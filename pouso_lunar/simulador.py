from dataclasses import dataclass
import math

import cv2
import numpy as np

from . import config


@dataclass
class Verdade:
    # onde a imagem foi tirada de verdade (para conferir o resultado depois)
    x: float
    y: float
    rumo_graus: float
    zoom: float
    M: np.ndarray


def centro_frame(tam: int) -> float:
    return (tam - 1) / 2.0


def meia_janela(zoom: float, tam: int) -> float:
    # metade da diagonal da área que a câmera vê, em pixels do mapa
    return tam / (2.0 * zoom) * math.sqrt(2.0)


def limitar_centro(x, y, zoom, tam, largura, altura) -> tuple[float, float]:
    # não deixa a câmera sair do mapa
    m = meia_janela(zoom, tam)
    x = largura / 2.0 if 2 * m >= largura else min(max(x, m), largura - m)
    y = altura / 2.0 if 2 * m >= altura else min(max(y, m), altura - m)
    return float(x), float(y)


def posicao_aleatoria(rng: np.random.Generator, zoom: float, tam: int,
                      largura: int, altura: int) -> tuple[float, float]:
    # sorteia uma posição onde a imagem inteira cabe no mapa
    m = meia_janela(zoom, tam)
    if 2 * m >= largura or 2 * m >= altura:
        return limitar_centro(0, 0, zoom, tam, largura, altura)
    return float(rng.uniform(m, largura - m)), float(rng.uniform(m, altura - m))


def gerar_frame(mapa: np.ndarray, x, y, rumo_graus, zoom, tam: int | None = None,
                ruido_sigma: float = 0.0, blur_ksize: int = 0, brilho: float = 0.0,
                gama: float = 1.0, rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, Verdade]:
    # Simula a câmera da sonda: recorta o mapa em volta de (x, y), girado e
    # ampliado. zoom pequeno = sonda alta, zoom grande = sonda baixa.
    tam = tam or config.TAM_FRAME
    c = centro_frame(tam)
    M = cv2.getRotationMatrix2D((float(x), float(y)), float(rumo_graus), float(zoom))
    M[:, 2] += (c - x, c - y)  # coloca o ponto (x, y) no centro da imagem
    frame = cv2.warpAffine(mapa, M, (tam, tam), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

    # "suja" a imagem para parecer mais com uma câmera de verdade
    if gama != 1.0 or brilho or blur_ksize or ruido_sigma:
        f = frame.astype(np.float32)
        if gama != 1.0:
            f = 255.0 * (f / 255.0) ** gama
        f += brilho
        if blur_ksize and blur_ksize > 1:
            f = cv2.GaussianBlur(f, (blur_ksize | 1, blur_ksize | 1), 0)
        if ruido_sigma:
            rng = rng or np.random.default_rng(config.SEMENTE)
            f += rng.normal(0, ruido_sigma, f.shape).astype(np.float32)
        frame = np.clip(f, 0, 255).astype(np.uint8)

    return frame, Verdade(float(x), float(y), float(rumo_graus), float(zoom), M)


def direcao_no_frame(M: np.ndarray, direcao_mapa_graus: float) -> float:
    # Como a câmera gira junto com a sonda, a direção do sol na imagem também
    # gira. Aqui calculamos essa direção a partir da rotação usada.
    a = math.radians(direcao_mapa_graus)
    vx, vy = M[:, :2] @ np.array([math.cos(a), math.sin(a)])
    return math.degrees(math.atan2(vy, vx)) % 360.0


def amostra_aleatoria(mapa: np.ndarray, n: int, semente: int, faixa_zoom=(0.6, 1.6),
                      degradar: bool = True, tam: int | None = None
                      ) -> list[tuple[np.ndarray, Verdade]]:
    # gera n imagens em posições, ângulos e alturas sorteados (sempre as mesmas pela semente)
    tam = tam or config.TAM_FRAME
    rng = np.random.default_rng(semente)
    alt, larg = mapa.shape[:2]
    saida = []
    for _ in range(n):
        zoom = float(rng.uniform(*faixa_zoom))
        rumo = float(rng.uniform(0, 360))
        x, y = posicao_aleatoria(rng, zoom, tam, larg, alt)
        deg = {}
        if degradar:
            deg = dict(ruido_sigma=float(rng.uniform(0, 10)),
                       blur_ksize=int(rng.choice([0, 3])),
                       brilho=float(rng.uniform(-25, 25)),
                       gama=float(rng.uniform(0.7, 1.4)))
        saida.append(gerar_frame(mapa, x, y, rumo, zoom, tam=tam, rng=rng, **deg))
    return saida


def trajetoria_descida(inicio: tuple[float, float], alvo: tuple[float, float],
                       n_frames: int = 240, zoom_ini: float = 0.5, zoom_fim: float = 1.6,
                       giro_total_graus: float = 45.0, *, largura: int, altura: int,
                       tam: int | None = None) -> list[tuple[float, float, float, float]]:
    # pontos (x, y, rumo, zoom) de uma descida do início até o alvo:
    # anda em linha reta, vai descendo (zoom aumenta) e gira um pouco
    tam = tam or config.TAM_FRAME
    traj = []
    for t in np.linspace(0.0, 1.0, n_frames):
        zoom = zoom_ini * (zoom_fim / zoom_ini) ** t
        x = inicio[0] + (alvo[0] - inicio[0]) * t
        y = inicio[1] + (alvo[1] - inicio[1]) * t
        x, y = limitar_centro(x, y, zoom, tam, largura, altura)
        traj.append((x, y, float(giro_total_graus * t), float(zoom)))
    return traj
