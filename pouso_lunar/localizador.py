"""Localização da sonda no mapa (features + homografia; template matching como comparação).

Ideia: achar no mapa os mesmos pontos característicos (keypoints) que a câmera
está vendo. Com ≥ 4 correspondências calcula-se a homografia frame → mapa;
o centro do frame projetado por ela é a posição da sonda, e a rotação dela
é o rumo. RANSAC descarta as correspondências erradas (outliers).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from . import config
from .simulador import centro_frame


@dataclass
class Resultado:
    x: float
    y: float
    rumo_graus: float
    n_inliers: int
    ok: bool
    H: np.ndarray | None = None


def _falha(n_inliers: int = 0) -> Resultado:
    return Resultado(float("nan"), float("nan"), float("nan"), int(n_inliers), False, None)


def homografia_valida(H) -> bool:
    """Rejeita homografias ausentes, com NaN, degeneradas ou espelhadas.

    det da parte linear = fator de área. Fora de [0,1; 10] o frame "encolheu"
    ou "explodiu" demais para ser uma vista real do terreno; det negativo é
    uma imagem espelhada, que uma câmera apontada para baixo não produz.
    """
    if H is None:
        return False
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    return bool(0.1 <= det <= 10.0)


def _criar_detector(metodo: str, para_mapa: bool):
    if metodo == "orb":
        n = config.ORB_FEATURES_MAPA if para_mapa else config.ORB_FEATURES_FRAME
        return cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING
    if metodo == "sift":
        return cv2.SIFT_create(), cv2.NORM_L2
    raise ValueError(f"método desconhecido: {metodo!r} (use 'orb' ou 'sift')")


class Localizador:
    """Localizador por features + homografia.

    `equalizar` (padrão: config.EQUALIZAR_LOCALIZACAO) aplica a equalização
    global do histograma ao mapa E ao frame. O mosaico é escuro (média
    ~38 em 255) e a câmera pode ver o terreno com outro brilho/contraste;
    equalizar os dois põe as intensidades na mesma escala antes das features.
    Com gama 1,5 o ORB passa de 27% para 100% de sucesso (notebook 03). O CLAHE
    também foi testado e piorou (realça o ruído local).
    """

    def __init__(self, mapa: np.ndarray, metodo: Literal["orb", "sift"] = "orb",
                 cache: Path | None = None, equalizar: bool | None = None):
        self.metodo = metodo
        self.equalizar = config.EQUALIZAR_LOCALIZACAO if equalizar is None else equalizar
        self.det_mapa, self.norma = _criar_detector(metodo, para_mapa=True)
        self.det_frame, _ = _criar_detector(metodo, para_mapa=False)
        if metodo == "sift":  # FLANN com KD-tree: busca aproximada, muito mais rápida em 70 mil pontos
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
        else:  # descritor binário: força bruta com distância de Hamming
            self.matcher = cv2.BFMatcher(self.norma)
        self.pts, self.desc = self._features_mapa(self._prep(mapa), Path(cache) if cache else None)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(img) if self.equalizar else img

    def _features_mapa(self, mapa, cache):
        # O cache só vale para o mesmo mapa e os mesmos parâmetros de extração.
        assinatura = f"{self.metodo}|{config.ORB_FEATURES_MAPA}|{self.equalizar}|{mapa.shape[:2]}"
        if cache and cache.exists():
            d = np.load(cache)
            if "assinatura" in d and str(d["assinatura"]) == assinatura:
                return d["pts"], d["desc"]
        kps, desc = self.det_mapa.detectAndCompute(mapa, None)
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
        if desc is None:
            desc = np.zeros((0, 32), np.uint8)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, pts=pts, desc=desc, assinatura=np.array(assinatura))
        return pts, desc

    def localizar(self, frame: np.ndarray,
                  janela: tuple[int, int, int, int] | None = None) -> Resultado:
        """Posição (px do mapa) e rumo da sonda. `janela`=(x0, y0, x1, y1) restringe a busca."""
        pts_m, desc_m = self.pts, self.desc
        if janela is not None:
            x0, y0, x1, y1 = janela
            sel = (pts_m[:, 0] >= x0) & (pts_m[:, 0] < x1) & (pts_m[:, 1] >= y0) & (pts_m[:, 1] < y1)
            pts_m, desc_m = pts_m[sel], desc_m[sel]
        if len(desc_m) < config.MIN_INLIERS:
            return _falha()

        kps, desc_f = self.det_frame.detectAndCompute(self._prep(frame), None)
        if desc_f is None or len(kps) < config.MIN_INLIERS:
            return _falha()

        # 2 vizinhos mais próximos + teste da razão de Lowe
        bons = [m[0] for m in self.matcher.knnMatch(desc_f, desc_m, k=2)
                if len(m) == 2 and m[0].distance < config.RAZAO_LOWE * m[1].distance]
        if len(bons) < config.MIN_INLIERS:
            return _falha(len(bons))

        src = np.float32([kps[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
        dst = pts_m[[m.trainIdx for m in bons]].reshape(-1, 1, 2)
        H, mascara = cv2.findHomography(src, dst, cv2.RANSAC, config.LIMIAR_RANSAC_PX)
        n_in = int(mascara.sum()) if mascara is not None else 0
        if n_in < config.MIN_INLIERS or not homografia_valida(H):
            return _falha(n_in)

        c = centro_frame(frame.shape[1])
        cx, cy = cv2.perspectiveTransform(np.float32([[[c, c]]]), H)[0, 0]
        # O frame é o mapa girado de +θ; a homografia frame→mapa desfaz essa
        # rotação, e o ângulo da sua parte linear é o próprio θ.
        rumo = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])) % 360.0)
        return Resultado(float(cx), float(cy), rumo, n_in, True, H)


def localizar_template(mapa: np.ndarray, frame: np.ndarray, zoom: float) -> Resultado:
    """Baseline: procura o frame inteiro no mapa por correlação.

    Precisa saber o zoom (altímetro) para voltar o frame à escala do mapa e não
    lida com rotação: é justamente essa limitação que a comparação mostra.
    """
    if zoom != 1.0:
        lado = max(8, round(frame.shape[1] / zoom))
        frame = cv2.resize(frame, (lado, lado), interpolation=cv2.INTER_AREA)
    if frame.shape[0] > mapa.shape[0] or frame.shape[1] > mapa.shape[1]:
        return _falha()
    res = cv2.matchTemplate(mapa, frame, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    c = centro_frame(frame.shape[1])
    ok = bool(max_val >= config.LIMIAR_TEMPLATE)
    return Resultado(float(max_loc[0] + c), float(max_loc[1] + c), 0.0, 0, ok, None)
