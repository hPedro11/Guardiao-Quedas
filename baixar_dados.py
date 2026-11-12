"""Baixa os dados públicos usados no projeto para a pasta dados/.

1. Recorte da região de estudo do mosaico global LROC WAC (NASA/LRO), lido
   por requisições HTTP Range: só ~37 MB dos 5,96 GB do arquivo global.
2. Catálogo de crateras de Robbins (2018), filtrado para a região de estudo.

Uso:
    python baixar_dados.py
"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd

from pouso_lunar import config
from pouso_lunar.geo import normalizar_lon

AGENTE = {"User-Agent": "pouso-lunar-gs-fiap/1.0"}
TENTATIVAS = 4


# ------------------------------------------------------------------ HTTP

def _abrir(url: str, inicio: int | None = None, fim: int | None = None):
    cab = dict(AGENTE)
    if inicio is not None:
        cab["Range"] = f"bytes={inicio}-{fim}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=cab), timeout=60)


def _ler_intervalo(url: str, inicio: int, fim: int) -> bytes:
    """Lê os bytes [inicio, fim] (inclusive), com novas tentativas."""
    for tentativa in range(TENTATIVAS):
        try:
            with _abrir(url, inicio, fim) as r:
                dados = r.read()
            if len(dados) != fim - inicio + 1:
                raise IOError(f"esperava {fim - inicio + 1} bytes, recebi {len(dados)}")
            return dados
        except Exception:
            if tentativa == TENTATIVAS - 1:
                raise
            time.sleep(1.5 * (tentativa + 1))
    raise AssertionError("inalcançável")


def urllib_fetch(url: str, arquivo) -> None:
    """Copia o conteúdo de `url` para o arquivo aberto, mostrando o progresso."""
    with _abrir(url) as r:
        total = int(r.headers.get("Content-Length") or 0)
        lido = 0
        while bloco := r.read(1 << 20):
            arquivo.write(bloco)
            lido += len(bloco)
            if total:
                print(f"\r  {lido / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    # Se a conexão cair, a leitura termina "normalmente", só que mais cedo.
    if total and lido != total:
        raise IOError(f"download incompleto: {lido} de {total} bytes")


def baixar_arquivo(url: str, destino: Path, buscar=urllib_fetch) -> None:
    """Baixa para `<destino>.part` e só renomeia quando terminar.

    Se a conexão cair no meio, o arquivo final não existe: um download
    corrompido nunca é confundido com um completo.
    """
    destino = Path(destino)
    parcial = destino.with_suffix(destino.suffix + ".part")
    with open(parcial, "wb") as f:
        buscar(url, f)
    os.replace(parcial, destino)


# --------------------------------------------------------------- mosaico

def _url_final(url: str) -> str:
    """Resolve o redirecionamento uma vez (planetarymaps → S3)."""
    with _abrir(url, 0, 0) as r:
        return r.geturl()


def baixar_regiao_mosaico(destino_png: Path, destino_json: Path) -> None:
    """Baixa só as linhas/colunas da região de estudo do BigTIFF global.

    O arquivo não tem compressão e guarda 1 linha por strip; a tabela
    StripOffsets diz onde cada linha começa. Então o pixel (linha r, coluna c)
    está no byte offset[r] + c, e cada linha do recorte é uma requisição Range.
    """
    ppg = config.PX_POR_GRAU_NATIVO
    r0 = math.floor((90.0 - config.LAT_MAX) * ppg)
    r1 = math.ceil((90.0 - config.LAT_MIN) * ppg)
    c0 = math.floor((config.LON_MIN + 180.0) * ppg)
    c1 = math.ceil((config.LON_MAX + 180.0) * ppg)
    altura, largura = r1 - r0, c1 - c0
    print(f"Mosaico: recorte de {largura} x {altura} px (linhas {r0}-{r1}, colunas {c0}-{c1})")

    url = _url_final(config.URL_MOSAICO)
    tab = config.MOSAICO_OFFSET_TABELA_STRIPS
    bruto = _ler_intervalo(url, tab + 8 * r0, tab + 8 * r1 - 1)
    offsets = struct.unpack(f"<{altura}Q", bruto)

    img = np.zeros((altura, largura), dtype=np.uint8)

    def baixar_linha(i: int) -> int:
        ini = offsets[i] + c0
        img[i] = np.frombuffer(_ler_intervalo(url, ini, ini + largura - 1), dtype=np.uint8)
        return i

    with ThreadPoolExecutor(16) as pool:
        for n, _ in enumerate(pool.map(baixar_linha, range(altura)), 1):
            if n % 250 == 0 or n == altura:
                print(f"\r  linhas {n}/{altura}", end="", flush=True)
    print()

    temp = destino_png.with_name(destino_png.stem + ".tmp.png")
    if not cv2.imwrite(str(temp), img):
        raise IOError(f"não consegui gravar {temp}")
    os.replace(temp, destino_png)

    info = {
        "lat_max": 90.0 - r0 / ppg,
        "lon_min": -180.0 + c0 / ppg,
        "px_por_grau": ppg,
        "largura": largura,
        "altura": altura,
    }
    destino_json.write_text(json.dumps(info, indent=2), encoding="utf-8")


# -------------------------------------------------------------- catálogo

def gravar_csv_atomico(df: pd.DataFrame, destino: Path) -> None:
    """Grava em <destino>.tmp e renomeia: um CSV truncado nunca fica com o nome final
    (senão a próxima execução diria "já existe, pulando" e manteria o arquivo quebrado)."""
    destino = Path(destino)
    temp = destino.with_suffix(destino.suffix + ".tmp")
    try:
        df.to_csv(temp, index=False)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    os.replace(temp, destino)


def filtrar_catalogo(df: pd.DataFrame, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame:
    """Normaliza a longitude e mantém as crateras da região (com 1° de margem).

    A margem garante que crateras cortadas pela borda do mapa ainda entrem
    no gabarito; o corte exato acontece em dados.carregar_gabarito().
    """
    lat = df["LAT_CIRC_IMG"].astype(float)
    lon = normalizar_lon(df["LON_CIRC_IMG"].astype(float).to_numpy())
    m = 1.0
    dentro = (
        (lat >= lat_min - m) & (lat <= lat_max + m)
        & (lon >= lon_min - m) & (lon <= lon_max + m)
    )
    return pd.DataFrame({
        "lat": lat[dentro].to_numpy(),
        "lon": lon[dentro.to_numpy()],
        "diam_km": df["DIAM_CIRC_IMG"].astype(float)[dentro].to_numpy(),
    })


def baixar_catalogo(destino_csv: Path) -> int:
    zip_path = destino_csv.parent / "robbins.zip"
    baixar_arquivo(config.URL_CATALOGO, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        csvs = [i for i in z.infolist() if i.filename.lower().endswith(".csv")]
        if not csvs:
            raise IOError("o zip do catálogo não tem CSV")
        maior = max(csvs, key=lambda i: i.file_size)
        print(f"  lendo {maior.filename}")
        with z.open(maior) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"),
                             usecols=["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"])
    regiao = filtrar_catalogo(df, config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX)
    gravar_csv_atomico(regiao, destino_csv)
    zip_path.unlink()
    return len(regiao)


# ----------------------------------------------------------------- main

def main() -> int:
    pasta = config.PASTA_DADOS
    pasta.mkdir(exist_ok=True)

    png, js = pasta / "mapa_regiao.png", pasta / "mapa_regiao.json"
    if png.exists() and js.exists():
        print(f"Mosaico já existe em {png}, pulando.")
    else:
        baixar_regiao_mosaico(png, js)

    csv = pasta / "crateras_regiao.csv"
    if csv.exists():
        print(f"Catálogo já existe em {csv}, pulando.")
    else:
        print("Catálogo de Robbins (96 MB):")
        n = baixar_catalogo(csv)
        print(f"  {n} crateras na região de estudo")

    info = json.loads(js.read_text(encoding="utf-8"))
    print(f"\nPronto: mapa {info['largura']} x {info['altura']} px, "
          f"{len(pd.read_csv(csv))} crateras no catálogo da região.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
