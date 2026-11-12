"""Testes offline do projeto: rodam sem rede e sem a pasta dados/.

Execução:
    python -m tests.test_offline
    python -m pytest tests -q
"""

import cv2
import numpy as np

from pouso_lunar.geo import MapaInfo, latlon_para_px, normalizar_lon, px_para_latlon

INFO = MapaInfo(lat_max=15.0, lon_min=15.0, px_por_grau=100.0, largura=2000, altura=2000)


# ---------------------------------------------------------------- geo (Task 1)

def test_latlon_px_ida_e_volta():
    x, y = latlon_para_px(5.0, 25.0, INFO)
    assert (x, y) == (1000.0, 1000.0)
    lat, lon = px_para_latlon(x, y, INFO)
    assert abs(lat - 5.0) < 1e-9 and abs(lon - 25.0) < 1e-9


def test_normalizar_lon():
    assert normalizar_lon(345.0) == -15.0
    assert normalizar_lon(25.0) == 25.0


def test_km_por_px():
    assert abs(INFO.km_por_px - 0.30323) < 1e-4


# ------------------------------------------------- download e dados (Task 2)

def test_filtrar_catalogo_normaliza_e_recorta():
    import pandas as pd

    from baixar_dados import filtrar_catalogo

    df = pd.DataFrame({
        "LAT_CIRC_IMG": [0.0, 0.0, 0.0],
        "LON_CIRC_IMG": [20.0, 345.0, 200.0],
        "DIAM_CIRC_IMG": [5.0, 5.0, 5.0],
    })
    r = filtrar_catalogo(df, -5.0, 15.0, 15.0, 35.0)
    assert list(r.columns) == ["lat", "lon", "diam_km"]
    assert len(r) == 1
    assert r["lon"].iloc[0] == 20.0


def test_download_interrompido_nao_deixa_arquivo_final(tmp_path):
    import pytest

    from baixar_dados import baixar_arquivo

    def buscar_falha(url, arquivo):
        arquivo.write(b"0123456789")
        raise IOError("conexão caiu")

    destino = tmp_path / "x.zip"
    with pytest.raises(IOError):
        baixar_arquivo("http://exemplo/x.zip", destino, buscar=buscar_falha)
    assert not destino.exists()


def test_download_curto_e_detectado(monkeypatch):
    import io

    import pytest

    import baixar_dados

    class RespostaCurta(io.BytesIO):
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(baixar_dados, "_abrir", lambda url, *a: RespostaCurta(b"x" * 10))
    with pytest.raises(IOError):
        baixar_dados.urllib_fetch("http://exemplo/x", io.BytesIO())


def test_gravar_csv_atomico_interrompido_nao_deixa_arquivo(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    from baixar_dados import gravar_csv_atomico

    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: (_ for _ in ()).throw(IOError("disco")))
    with pytest.raises(IOError):
        gravar_csv_atomico(df, tmp_path / "c.csv")
    assert not (tmp_path / "c.csv").exists()


def test_carregar_mapa_sem_dados_explica(tmp_path, monkeypatch):
    import pytest

    from pouso_lunar import config, dados

    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path)
    with pytest.raises(FileNotFoundError, match="baixar_dados.py"):
        dados.carregar_mapa()


def test_carregar_gabarito_converte_raio(tmp_path, monkeypatch):
    import pandas as pd

    from pouso_lunar import config, dados

    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path)
    pd.DataFrame({"lat": [5.0], "lon": [25.0], "diam_km": [6.0647]}).to_csv(
        tmp_path / "crateras_regiao.csv", index=False
    )
    gab = dados.carregar_gabarito(INFO, diam_min_px=10)
    assert len(gab) == 1
    assert abs(gab["r_px"].iloc[0] - 10.0) < 0.01
    assert abs(gab["x"].iloc[0] - 1000.0) < 1e-6 and abs(gab["y"].iloc[0] - 1000.0) < 1e-6


def test_carregar_gabarito_descarta_bacias_gigantes(tmp_path, monkeypatch):
    import pandas as pd

    from pouso_lunar import config, dados

    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path)
    pd.DataFrame({"lat": [5.0, 5.0], "lon": [25.0, 25.0], "diam_km": [6.0, 900.0]}).to_csv(
        tmp_path / "crateras_regiao.csv", index=False
    )
    gab = dados.carregar_gabarito(INFO, diam_min_px=10, diam_max_px=120)
    assert list(gab["diam_km"]) == [6.0]


# ------------------------------------------------------- detector (Task 3)

def test_detector_acha_crateras_sinteticas():
    from pouso_lunar.detector import ParamsDeteccao, detectar_crateras
    from pouso_lunar.metricas import casar_deteccoes
    from tests.sintetico import cena_crateras

    img, gab = cena_crateras(42)
    det = detectar_crateras(img, ParamsDeteccao(metodo="hough", usar_clahe=True, blur_ksize=5,
                                                r_min=6, r_max=35))
    m = casar_deteccoes(det, gab)
    assert m["tp"] / len(gab) >= 0.8


def test_detector_sombra_acha_crateras_sinteticas():
    from pouso_lunar.detector import ParamsDeteccao, detectar_crateras
    from pouso_lunar.metricas import casar_deteccoes, prf1
    from tests.sintetico import cena_crateras

    img, gab = cena_crateras(42, amplitude=6)  # rugosidade entre mar (3) e terras altas (10)
    det = detectar_crateras(img, ParamsDeteccao(metodo="sombra"))
    m = casar_deteccoes(det, gab)
    p, r, _ = prf1(m["tp"], m["fp"], m["fn"])
    assert r >= 0.8 and p >= 0.5


def test_detector_sombra_respeita_direcao_do_sol():
    from pouso_lunar.detector import ParamsDeteccao, detectar_crateras
    from pouso_lunar.metricas import casar_deteccoes
    from tests.sintetico import cena_crateras

    img, gab = cena_crateras(42)
    girada = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # sol passa a vir do norte
    gab_g = np.column_stack([img.shape[0] - 1 - gab[:, 1], gab[:, 0], gab[:, 2]])
    certo = casar_deteccoes(detectar_crateras(girada, ParamsDeteccao(metodo="sombra", direcao_sol_graus=90)), gab_g)
    errado = casar_deteccoes(detectar_crateras(girada, ParamsDeteccao(metodo="sombra", direcao_sol_graus=0)), gab_g)
    assert certo["tp"] / len(gab) >= 0.8
    assert errado["tp"] < certo["tp"] / 2


def test_sombra_nao_pareia_manchas_distantes():
    from pouso_lunar.detector import ParamsDeteccao, detectar_crateras

    img = np.full((200, 200), 120, np.uint8)
    cv2.circle(img, (100, 100), 4, 40, -1)   # sombra (área ~50 px)
    cv2.circle(img, (125, 100), 4, 220, -1)  # brilho a 25 px, a leste
    assert len(detectar_crateras(img, ParamsDeteccao())) == 0  # 25 > 2,5·√50 + 3
    assert len(detectar_crateras(img, ParamsDeteccao(alcance_fator=4.0))) == 1


def test_suprimir_sobrepostos_rapido():
    import time

    from pouso_lunar.detector import suprimir_sobrepostos

    rng = np.random.default_rng(0)
    c = np.column_stack([rng.uniform(0, 3000, 15000), rng.uniform(0, 3000, 15000),
                         rng.uniform(5, 60, 15000)]).astype(np.float32)
    t0 = time.perf_counter()
    suprimir_sobrepostos(c)
    # folga para máquinas lentas; a versão O(n²) em Python levava ~150 s
    assert time.perf_counter() - t0 < 30.0


def test_detector_imagem_lisa_nao_inventa():
    from pouso_lunar.detector import detectar_crateras

    det = detectar_crateras(np.zeros((512, 512), np.uint8))
    assert det.shape == (0, 3)


def test_suprimir_sobrepostos():
    from pouso_lunar.detector import suprimir_sobrepostos

    c = np.array([[100, 100, 20], [105, 100, 18], [300, 300, 10]], np.float32)
    assert len(suprimir_sobrepostos(c)) == 2


# -------------------------------------------------------- métricas (Task 4)

def test_casar_um_para_um():
    from pouso_lunar.metricas import casar_deteccoes

    m = casar_deteccoes(np.array([[0, 0, 10], [1, 0, 10]], float), np.array([[0, 0, 10]], float))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_casar_raio_fora_da_faixa():
    from pouso_lunar.metricas import casar_deteccoes

    m = casar_deteccoes(np.array([[0, 0, 30]], float), np.array([[0, 0, 10]], float))
    assert m["tp"] == 0


def test_casar_listas_vazias():
    from pouso_lunar.metricas import casar_deteccoes

    m = casar_deteccoes(np.zeros((0, 3)), np.array([[0, 0, 10]], float))
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 1)


def test_prf1_zero():
    from pouso_lunar.metricas import prf1

    assert prf1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_erro_km():
    from pouso_lunar.metricas import erro_km

    assert erro_km(3, 4, 0, 0, 0.2) == 1.0


# ------------------------------------------------------- simulador (Task 5)

def test_gerar_frame_centro_vai_para_o_meio():
    from pouso_lunar.simulador import gerar_frame

    mapa = np.zeros((600, 600), np.uint8)
    mapa[200, 300] = 255
    frame, verdade = gerar_frame(mapa, 300, 200, 30, 1.0, tam=101)
    y, x = np.unravel_index(np.argmax(frame), frame.shape)
    assert abs(x - 50) <= 1 and abs(y - 50) <= 1
    assert verdade.x == 300 and verdade.y == 200 and verdade.M.shape == (2, 3)


def test_trajetoria_respeita_bordas():
    from pouso_lunar.simulador import meia_janela, trajetoria_descida

    traj = trajetoria_descida((1500, 1500), (0, 0), largura=3000, altura=3000)
    assert len(traj) == 240
    for x, y, rumo, zoom in traj:
        m = meia_janela(zoom, 480)
        assert m <= x <= 3000 - m and m <= y <= 3000 - m


def test_direcao_do_sol_gira_com_o_frame():
    from pouso_lunar.simulador import direcao_no_frame, gerar_frame

    mapa = np.zeros((600, 600), np.uint8)
    _, v0 = gerar_frame(mapa, 300, 300, 0, 1.0, tam=101)
    _, v90 = gerar_frame(mapa, 300, 300, 90, 1.0, tam=101)
    assert abs(direcao_no_frame(v0.M, 0.0) - 0.0) < 1e-6
    assert abs(direcao_no_frame(v90.M, 0.0) - 270.0) < 1e-6


def test_posicoes_aleatorias_nao_se_acumulam_na_borda():
    from pouso_lunar.simulador import meia_janela, posicao_aleatoria

    rng = np.random.default_rng(0)
    m = meia_janela(0.5, 480)
    pts = np.array([posicao_aleatoria(rng, 0.5, 480, 3000, 3000) for _ in range(2000)])
    assert pts.min() >= m and pts.max() <= 3000 - m
    na_borda = np.isclose(pts, m) | np.isclose(pts, 3000 - m)
    assert na_borda.mean() < 0.01


def test_amostra_reprodutivel():
    from pouso_lunar.simulador import amostra_aleatoria
    from tests.sintetico import textura

    mapa = np.clip(textura(1, 1500), 0, 255).astype(np.uint8)
    a = [v for _, v in amostra_aleatoria(mapa, 5, semente=7)]
    b = [v for _, v in amostra_aleatoria(mapa, 5, semente=7)]
    assert [(v.x, v.y, v.rumo_graus, v.zoom) for v in a] == [(v.x, v.y, v.rumo_graus, v.zoom) for v in b]


# ------------------------------------------------------ localizador (Task 6)

_MAPA = None


def _mapa():
    global _MAPA
    if _MAPA is None:
        from tests.sintetico import mapa_sintetico
        _MAPA = mapa_sintetico(1500)
    return _MAPA


def _dif_angular(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_localiza_frame_conhecido():
    from pouso_lunar.localizador import Localizador
    from pouso_lunar.simulador import gerar_frame

    frame, v = gerar_frame(_mapa(), 700, 800, 40, 1.2, tam=320)
    r = Localizador(_mapa(), "orb").localizar(frame)
    assert r.ok
    assert np.hypot(r.x - 700, r.y - 800) < 3


def test_rumo_estimado():
    from pouso_lunar.localizador import Localizador
    from pouso_lunar.simulador import gerar_frame

    frame, _ = gerar_frame(_mapa(), 700, 800, 40, 1.2, tam=320)
    r = Localizador(_mapa(), "orb").localizar(frame)
    assert _dif_angular(r.rumo_graus, 40) < 2


def test_sift_tambem_localiza():
    from pouso_lunar.localizador import Localizador
    from pouso_lunar.simulador import gerar_frame

    frame, _ = gerar_frame(_mapa(), 900, 600, 200, 0.8, tam=320)
    r = Localizador(_mapa(), "sift").localizar(frame)
    assert r.ok and np.hypot(r.x - 900, r.y - 600) < 3
    assert _dif_angular(r.rumo_graus, 200) < 2


def test_localiza_frame_escurecido_em_mapa_escuro():
    from pouso_lunar.localizador import Localizador
    from pouso_lunar.simulador import gerar_frame

    escuro = (_mapa() // 3).astype(np.uint8)  # média ~40, como o mosaico real
    frame, _ = gerar_frame(escuro, 700, 800, 25, 1.0, tam=320, gama=2.0)
    r = Localizador(escuro, "orb").localizar(frame)
    assert r.ok and np.hypot(r.x - 700, r.y - 800) < 3


def test_frame_liso_nao_quebra():
    from pouso_lunar.localizador import Localizador

    r = Localizador(_mapa(), "orb").localizar(np.full((320, 320), 128, np.uint8))
    assert r.ok is False


def test_homografia_valida():
    from pouso_lunar.localizador import homografia_valida

    assert homografia_valida(None) is False
    assert homografia_valida(np.full((3, 3), np.nan)) is False
    assert homografia_valida(np.diag([0.01, 0.01, 1.0])) is False
    assert homografia_valida(np.diag([-1.0, 1.0, 1.0])) is False  # espelhada: câmera não faz isso
    assert homografia_valida(np.eye(3)) is True


def test_cache_invalida_quando_parametros_mudam(tmp_path, monkeypatch):
    from pouso_lunar import config
    from pouso_lunar.localizador import Localizador

    cache = tmp_path / "orb.npz"
    monkeypatch.setattr(config, "ORB_FEATURES_MAPA", 500)
    assert len(Localizador(_mapa(), "orb", cache=cache).pts) <= 500
    monkeypatch.setattr(config, "ORB_FEATURES_MAPA", 2000)
    assert len(Localizador(_mapa(), "orb", cache=cache).pts) > 500


def test_template_sem_rotacao():
    from pouso_lunar.localizador import localizar_template
    from pouso_lunar.simulador import gerar_frame

    frame, _ = gerar_frame(_mapa(), 650, 720, 0, 1.0, tam=320)
    r = localizar_template(_mapa(), frame, zoom=1.0)
    assert r.ok and np.hypot(r.x - 650, r.y - 720) < 3


def test_localizar_com_janela_e_cache(tmp_path):
    from pouso_lunar.localizador import Localizador
    from pouso_lunar.simulador import gerar_frame

    cache = tmp_path / "orb.npz"
    Localizador(_mapa(), "orb", cache=cache)
    assert cache.exists()
    loc = Localizador(_mapa(), "orb", cache=cache)
    frame, _ = gerar_frame(_mapa(), 700, 800, 10, 1.0, tam=320)
    r = loc.localizar(frame, janela=(400, 500, 1000, 1100))
    assert r.ok and np.hypot(r.x - 700, r.y - 800) < 3


# ---------------------------------------------------------- visual (Task 8)

def test_minimapa_ida_e_volta():
    from pouso_lunar.visual import Minimapa

    mini = Minimapa(np.zeros((3000, 3000), np.uint8), 480)
    x, y = mini.para_mapa(*mini.para_mini(1500, 750))
    assert np.hypot(x - 1500, y - 750) < 7


def test_montar_painel_formato():
    from pouso_lunar.visual import montar_painel

    frame = np.zeros((480, 480, 3), np.uint8)
    mini = np.zeros((480, 480, 3), np.uint8)
    assert montar_painel(frame, mini, ["crateras: 3"]).shape == (480, 960, 3)


def test_demo_alvo_clicado_na_borda_fica_alcancavel(tmp_path, monkeypatch):
    import cv2 as _cv2

    from main import Demo
    from pouso_lunar import config

    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path)
    info = MapaInfo(lat_max=15.0, lon_min=15.0, px_por_grau=100.0, largura=1500, altura=1500)
    demo = Demo(_mapa(), info, "orb", 1)
    demo.ao_clicar(_cv2.EVENT_LBUTTONDOWN, demo.tam + 479, 479, 0, None)  # canto do minimapa
    demo.passo()
    fim = demo.traj[-1][:2]
    assert np.hypot(demo.alvo[0] - fim[0], demo.alvo[1] - fim[1]) < 1e-6


def test_main_sem_dados_sai_limpo(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    shutil.copy(raiz / "main.py", tmp_path / "main.py")
    shutil.copytree(raiz / "pouso_lunar", tmp_path / "pouso_lunar",
                    ignore=shutil.ignore_patterns("__pycache__"))
    p = subprocess.run([sys.executable, "main.py"], cwd=tmp_path, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=60)
    saida = p.stdout + p.stderr
    assert p.returncode == 1
    assert "baixar_dados.py" in saida
    assert "Traceback" not in saida


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
