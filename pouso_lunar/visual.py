import cv2
import numpy as np

# cores em BGR (o OpenCV usa azul, verde, vermelho)
VERDE = (0, 200, 0)
VERMELHO = (0, 0, 255)
CIANO = (255, 255, 0)
AMARELO = (0, 255, 255)
BRANCO = (255, 255, 255)


def em_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def desenhar_crateras(img_bgr: np.ndarray, circulos, cor=AMARELO) -> np.ndarray:
    saida = em_bgr(img_bgr)
    for x, y, r in np.asarray(circulos).reshape(-1, 3):
        cv2.circle(saida, (int(round(x)), int(round(y))), int(round(r)), cor, 1, cv2.LINE_AA)
    return saida


def escrever(img: np.ndarray, texto: str, pos, cor=BRANCO, escala: float = 0.55) -> None:
    # a fonte do OpenCV não tem acento, por isso os textos da tela vão sem acento
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def caixa_escura(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alfa: float = 0.55) -> None:
    # escurece o fundo atrás do texto para dar para ler
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alfa)).astype(img.dtype)


def montar_painel(frame_bgr: np.ndarray, minimapa_bgr: np.ndarray, textos: list[str],
                  cores: list | None = None) -> np.ndarray:
    # câmera à esquerda, minimapa à direita e os textos no canto
    esq = em_bgr(frame_bgr)
    dir_ = em_bgr(minimapa_bgr)
    if dir_.shape[0] != esq.shape[0]:
        lado = round(dir_.shape[1] * esq.shape[0] / dir_.shape[0])
        dir_ = cv2.resize(dir_, (lado, esq.shape[0]), interpolation=cv2.INTER_AREA)
    painel = np.hstack([esq, dir_])
    if textos:
        caixa_escura(painel, 0, 0, 215, 12 + 22 * len(textos))
    for i, t in enumerate(textos):
        cor = cores[i] if cores and i < len(cores) else BRANCO
        escrever(painel, t, (10, 24 + 22 * i), cor)
    return painel


class Minimapa:
    # mapa em tamanho pequeno, com conversão de coordenadas entre ele e o mapa grande

    def __init__(self, mapa: np.ndarray, lado: int = 480):
        self.lado = lado
        self.sx = lado / mapa.shape[1]
        self.sy = lado / mapa.shape[0]
        self.base = em_bgr(cv2.resize(mapa, (lado, lado), interpolation=cv2.INTER_AREA))

    def para_mini(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * self.sx)), int(round(y * self.sy))

    def para_mapa(self, u: float, v: float) -> tuple[float, float]:
        return u / self.sx, v / self.sy

    def desenhar(self, traj_real, traj_est, alvo, quadro=None) -> np.ndarray:
        # caminho real em verde, caminho encontrado em vermelho, alvo em ciano
        # e o quadrado amarelo mostra o que a câmera está vendo
        img = self.base.copy()
        for traj, cor, esp in ((traj_real, VERDE, 4), (traj_est, VERMELHO, 1)):
            pts = [self.para_mini(x, y) for x, y in traj if np.isfinite(x) and np.isfinite(y)]
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32)], False, cor, esp, cv2.LINE_AA)
        if quadro is not None:
            q = np.array([self.para_mini(x, y) for x, y in quadro], np.int32)
            cv2.polylines(img, [q], True, AMARELO, 1, cv2.LINE_AA)
        if alvo is not None:
            cv2.drawMarker(img, self.para_mini(*alvo), CIANO, cv2.MARKER_CROSS, 14, 2)
        return img
