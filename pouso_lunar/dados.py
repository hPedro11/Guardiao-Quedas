import json
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from . import config
from .geo import MapaInfo, latlon_para_px

ARQ_MAPA = "mapa_regiao.png"
ARQ_INFO = "mapa_regiao.json"
ARQ_CRATERAS = "crateras_regiao.csv"


def _exigir(caminho):
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo {caminho} não encontrado. Baixe os dados antes: rode `python baixar_dados.py`."
        )


def carregar_mapa(escala: float | None = None) -> tuple[np.ndarray, MapaInfo]:
    escala = config.ESCALA_MAPA if escala is None else escala
    pasta = config.PASTA_DADOS
    p_img, p_info = pasta / ARQ_MAPA, pasta / ARQ_INFO
    _exigir(p_img)
    _exigir(p_info)

    img = cv2.imread(str(p_img), cv2.IMREAD_GRAYSCALE)
    meta = json.loads(p_info.read_text(encoding="utf-8"))
    if escala != 1.0:
        larg = round(img.shape[1] * escala)
        alt = round(img.shape[0] * escala)
        img = cv2.resize(img, (larg, alt), interpolation=cv2.INTER_AREA)
    # ao diminuir o mapa, os pixels por grau diminuem junto
    fator = img.shape[1] / meta["largura"]
    info = MapaInfo(
        lat_max=meta["lat_max"],
        lon_min=meta["lon_min"],
        px_por_grau=meta["px_por_grau"] * fator,
        largura=img.shape[1],
        altura=img.shape[0],
    )
    return img, info


def carregar_gabarito(info: MapaInfo, diam_min_px: float | None = None,
                      diam_max_px: float | None = None) -> pd.DataFrame:
    # crateras do catálogo convertidas para pixels do mapa (colunas x, y, r_px, diam_km)
    diam_min_px = config.DIAM_MIN_PX if diam_min_px is None else diam_min_px
    diam_max_px = config.DIAM_MAX_PX if diam_max_px is None else diam_max_px
    p = config.PASTA_DADOS / ARQ_CRATERAS
    _exigir(p)
    cat = pd.read_csv(p)
    x, y = latlon_para_px(cat["lat"].to_numpy(), cat["lon"].to_numpy(), info)
    r = cat["diam_km"].to_numpy() / 2.0 / info.km_por_px
    ok = ((x >= 0) & (x < info.largura) & (y >= 0) & (y < info.altura)
          & (2 * r >= diam_min_px) & (2 * r <= diam_max_px))
    return pd.DataFrame({
        "x": x[ok], "y": y[ok], "r_px": r[ok], "diam_km": cat["diam_km"].to_numpy()[ok],
    })


def metade(obj, info: MapaInfo, qual: Literal["ajuste", "teste"]):
    # Divide o mapa ao meio: a metade oeste ("ajuste") serve para escolher os
    # parâmetros e a leste ("teste") para conferir o resultado.
    xs = obj["x"].to_numpy() if isinstance(obj, pd.DataFrame) else np.asarray(obj)[:, 0]
    meio = info.largura / 2.0
    m = xs < meio if qual == "ajuste" else xs >= meio
    return obj[m].reset_index(drop=True) if isinstance(obj, pd.DataFrame) else np.asarray(obj)[m]
