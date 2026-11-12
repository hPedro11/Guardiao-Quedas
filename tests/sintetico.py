"""Cenas sintéticas para os testes offline (sem baixar dados)."""

import cv2
import numpy as np


def textura(semente: int, tam: int = 512, amplitude: float = 12.0) -> np.ndarray:
    """Fundo de ruído suavizado, parecido com regolito em baixa resolução.

    `amplitude` controla a rugosidade: 12 ≈ terras altas ásperas; no mapa real
    o desvio local é ~3 no mar e ~10 nas terras altas.
    """
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0, 1, (tam, tam)).astype(np.float32)
    suave = cv2.GaussianBlur(ruido, (0, 0), 3)
    suave = (suave - suave.mean()) / (suave.std() + 1e-6)
    return 120 + amplitude * suave


def mapa_sintetico(tam: int = 1500) -> np.ndarray:
    """Mapa com textura rica: mosaico de cenas de crateras com sementes diferentes."""
    blocos = -(-tam // 512)
    linhas = [np.hstack([cena_crateras(100 + i * blocos + j)[0] for j in range(blocos)])
              for i in range(blocos)]
    return np.vstack(linhas)[:tam, :tam].copy()


def cena_crateras(semente: int, n: int = 25, tam: int = 512,
                  amplitude: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Imagem uint8 com `n` crateras sombreadas (sol a oeste) e o gabarito (n, 3).

    Cada cratera: parede oeste em sombra, parede leste iluminada (gradiente
    horizontal dentro do disco) e um aro claro em volta. Raios de 8 a 30 px,
    sem sobreposição.
    """
    rng = np.random.default_rng(semente)
    img = textura(semente, tam, amplitude)
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float32)
    circulos: list[tuple[float, float, float]] = []
    tentativas = 0
    while len(circulos) < n and tentativas < 5000:
        tentativas += 1
        r = float(rng.uniform(8, 30))
        x = float(rng.uniform(r * 1.3, tam - r * 1.3))
        y = float(rng.uniform(r * 1.3, tam - r * 1.3))
        if any(np.hypot(x - cx, y - cy) < 1.3 * (r + cr) for cx, cy, cr in circulos):
            continue
        circulos.append((x, y, r))
        d = np.hypot(xx - x, yy - y)
        dentro = d < r
        img[dentro] = 120 + 70 * ((xx[dentro] - x) / r)
        aro = (d >= r) & (d < 1.15 * r)
        img[aro] += 25
    return np.clip(img, 0, 255).astype(np.uint8), np.array(circulos, dtype=np.float32)
