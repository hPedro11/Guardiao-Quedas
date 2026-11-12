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
    # Descarta homografias estranhas: vazias, com NaN, espelhadas (determinante
    # negativo) ou que encolhem/esticam demais a imagem.
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
    # Acha no mapa os mesmos pontos que a câmera está vendo e calcula a
    # homografia entre a imagem e o mapa. O centro da imagem levado para o
    # mapa é a posição da sonda.

    def __init__(self, mapa: np.ndarray, metodo: Literal["orb", "sift"] = "orb",
                 cache: Path | None = None, equalizar: bool | None = None):
        self.metodo = metodo
        self.equalizar = config.EQUALIZAR_LOCALIZACAO if equalizar is None else equalizar
        self.det_mapa, self.norma = _criar_detector(metodo, para_mapa=True)
        self.det_frame, _ = _criar_detector(metodo, para_mapa=False)
        if metodo == "sift":
            # o SIFT gera muitos pontos no mapa, então usamos o FLANN, que é mais rápido
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
        else:
            self.matcher = cv2.BFMatcher(self.norma)
        self.pts, self.desc = self._features_mapa(self._prep(mapa), Path(cache) if cache else None)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        # o mapa é bem escuro; equalizar o histograma deixa mapa e câmera parecidos
        return cv2.equalizeHist(img) if self.equalizar else img

    def _features_mapa(self, mapa, cache):
        # os pontos do mapa demoram para calcular, então salvamos num arquivo de cache
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
        # janela = (x0, y0, x1, y1): se passar, procura só nesse pedaço do mapa
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

        # para cada ponto pega os 2 mais parecidos do mapa e aplica o teste de Lowe
        bons = [m[0] for m in self.matcher.knnMatch(desc_f, desc_m, k=2)
                if len(m) == 2 and m[0].distance < config.RAZAO_LOWE * m[1].distance]
        if len(bons) < config.MIN_INLIERS:
            return _falha(len(bons))

        # homografia com RANSAC, que ignora os pares errados
        src = np.float32([kps[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
        dst = pts_m[[m.trainIdx for m in bons]].reshape(-1, 1, 2)
        H, mascara = cv2.findHomography(src, dst, cv2.RANSAC, config.LIMIAR_RANSAC_PX)
        n_in = int(mascara.sum()) if mascara is not None else 0
        if n_in < config.MIN_INLIERS or not homografia_valida(H):
            return _falha(n_in)

        # posição = onde o centro da imagem cai no mapa
        c = centro_frame(frame.shape[1])
        cx, cy = cv2.perspectiveTransform(np.float32([[[c, c]]]), H)[0, 0]
        # rumo = ângulo de rotação da homografia
        rumo = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])) % 360.0)
        return Resultado(float(cx), float(cy), rumo, n_in, True, H)


def localizar_template(mapa: np.ndarray, frame: np.ndarray, zoom: float) -> Resultado:
    # Comparação simples: procura a imagem inteira no mapa com matchTemplate.
    # Não funciona se a imagem estiver girada.
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
