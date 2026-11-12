"""Detecção de crateras com PDI clássico.

Dois métodos, comparados na ablação do notebook 02:

**sombra** (principal), com sol baixo:
    tons de cinza → blur gaussiano
    → limiar LOCAL: pixel muito mais escuro que a vizinhança = sombra,
      muito mais claro = brilho (limiarização)
    → abertura morfológica para tirar ruído
    → componentes conectados + centroides (contornos e momentos)
    → pareamento: cada sombra com o brilho mais próximo NA DIREÇÃO DO SOL
    → cratera: centro no meio do par, raio proporcional à distância entre eles

    Com o sol baixo, uma cratera pequena não aparece como um círculo fechado,
    e sim como uma mancha escura (a parede do lado do sol, em sombra) colada
    a uma mancha clara (a parede oposta, iluminada). Esse par é a assinatura
    da cratera.

**hough** (comparação):
    tons de cinza → CLAHE → blur → HoughCircles.
    Funciona para crateras grandes com borda nítida. Nas pequenas falta arco
    para votar, e na textura das terras altas ele encaixa círculos em arcos
    aleatórios (ver notebook 02).
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class ParamsDeteccao:
    metodo: str = "sombra"  # "sombra" ou "hough"

    # Pré-processamento
    # CLAHE equaliza o contraste por blocos. Ajuda o Hough, mas no
    # método da sombra amplifica a textura e gera pares falsos (ablação).
    usar_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grade: int = 8
    # Blur gaussiano: tira o ruído fino antes do limiar e do gradiente.
    blur_ksize: int = 3

    # Raios aceitos (px)
    r_min: int = 5
    r_max: int = 60

    # Método da sombra
    limiar_sombra: float = 18.0  # quanto (níveis de cinza) abaixo/acima da média local
    janela_media: int = 21  # lado da janela da média local (px): maior que as crateras pequenas
    area_min: int = 4  # manchas menores que isso são ruído
    direcao_sol_graus: float = 0.0  # direção sombra→brilho na imagem (0 = leste/+x, 90 = sul/+y)
    ang_max: float = 40.0  # tolerância angular do pareamento
    # Sombra e brilho são as duas paredes da MESMA cratera: ficam coladas
    # (distância ≤ alcance_fator·√área_da_sombra + 3) e têm áreas parecidas
    # (razão ≤ razao_area_max). Pares distantes ou desproporcionais são textura.
    alcance_fator: float = 2.5
    razao_area_max: float = 6.0
    fator_raio: float = 0.8  # raio = fator · distância entre os centroides do par

    # Método Hough (HOUGH_GRADIENT)
    dp: float = 1.2  # resolução do acumulador = imagem/dp
    min_dist_fator: float = 0.8  # minDist = fator · r_min
    canny_alto: float = 100  # param1: limiar alto do Canny interno
    acumulador: float = 25  # param2: votos mínimos (menor = mais detecções e mais falsos)

    # Validação opcional por contraste, usada só na ablação do Hough
    validar_contraste: bool = False
    contraste_min: float = 8.0


def preprocessar(img: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if p.usar_clahe:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(p.clahe_grade, p.clahe_grade))
        cinza = clahe.apply(cinza)
    if p.blur_ksize and p.blur_ksize > 1:
        k = p.blur_ksize | 1  # o kernel precisa ser ímpar
        cinza = cv2.GaussianBlur(cinza, (k, k), 0)
    return cinza


def suprimir_sobrepostos(c: np.ndarray) -> np.ndarray:
    """Remove o círculo posterior quando dois centros estão a < 0,5·min(r).

    A ordem de entrada é a de prioridade (o HoughCircles devolve do mais votado
    para o menos votado). Usa uma grade espacial: cada círculo só é comparado
    com os já mantidos nas células vizinhas, e não com todos.
    """
    c = np.asarray(c, dtype=np.float32).reshape(-1, 3)
    if len(c) == 0:
        return c
    lado = max(1.0, 0.5 * float(c[:, 2].max()))
    grade: dict[tuple[int, int], list[int]] = {}
    mantidos: list[int] = []
    for i, (x, y, r) in enumerate(c):
        gx, gy = int(x // lado), int(y // lado)
        conflito = False
        for vx in (gx - 1, gx, gx + 1):
            for vy in (gy - 1, gy, gy + 1):
                for j in grade.get((vx, vy), ()):
                    mx, my, mr = c[j]
                    if math.hypot(x - mx, y - my) < 0.5 * min(r, mr):
                        conflito = True
                        break
                if conflito:
                    break
            if conflito:
                break
        if not conflito:
            mantidos.append(i)
            grade.setdefault((gx, gy), []).append(i)
    return c[mantidos]


def _manchas(mascara: np.ndarray, area_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Centroides (N, 2) e áreas (N,) dos componentes conectados com área ≥ area_min."""
    _, _, stats, centros = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ok = areas >= area_min
    return centros[1:][ok].astype(np.float32), areas[ok].astype(np.float32)


def _detectar_sombra(proc: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    f = proc.astype(np.float32)
    media = cv2.blur(f, (p.janela_media, p.janela_media))
    sombra = np.uint8(f < media - p.limiar_sombra) * 255
    brilho = np.uint8(f > media + p.limiar_sombra) * 255
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sombra = cv2.morphologyEx(sombra, cv2.MORPH_OPEN, elem)
    brilho = cv2.morphologyEx(brilho, cv2.MORPH_OPEN, elem)

    cs, area_s = _manchas(sombra, p.area_min)
    cb, area_b = _manchas(brilho, p.area_min)
    if len(cs) == 0 or len(cb) == 0:
        return np.zeros((0, 3), np.float32)

    a = math.radians(p.direcao_sol_graus)
    d = np.array([math.cos(a), math.sin(a)], np.float32)
    usado = np.zeros(len(cb), bool)
    crateras = []
    for i in np.argsort(-area_s):  # sombras maiores primeiro
        v = cb - cs[i]
        dist = np.hypot(v[:, 0], v[:, 1])
        ao_longo = v @ d
        transversal = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
        ang = np.degrees(np.arctan2(transversal, ao_longo))
        alcance = p.alcance_fator * math.sqrt(float(area_s[i])) + 3.0
        razao = np.maximum(area_b / area_s[i], area_s[i] / area_b)
        cand = ((ang < p.ang_max) & (dist > 1.0) & (dist < alcance)
                & (razao <= p.razao_area_max) & ~usado)
        if not cand.any():
            continue
        j = np.flatnonzero(cand)[np.argmin(dist[cand])]
        usado[j] = True
        r = p.fator_raio * float(dist[j])
        if p.r_min <= r <= p.r_max:
            centro = (cs[i] + cb[j]) / 2.0
            crateras.append((centro[0], centro[1], r))
    return suprimir_sobrepostos(np.array(crateras, np.float32).reshape(-1, 3))


def _contraste(img: np.ndarray, x: float, y: float, r: float) -> float:
    """|média do anel (r..1,4r) − média do disco (0,7r)|."""
    disco = np.zeros(img.shape, np.uint8)
    anel = np.zeros(img.shape, np.uint8)
    centro = (int(round(x)), int(round(y)))
    cv2.circle(disco, centro, max(1, int(0.7 * r)), 255, -1)
    cv2.circle(anel, centro, int(1.4 * r), 255, -1)
    cv2.circle(anel, centro, int(r), 0, -1)
    if not anel.any() or not disco.any():
        return 0.0
    return abs(cv2.mean(img, anel)[0] - cv2.mean(img, disco)[0])


def _detectar_hough(proc: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    circ = cv2.HoughCircles(
        proc, cv2.HOUGH_GRADIENT, dp=p.dp,
        minDist=max(1.0, p.min_dist_fator * p.r_min),
        param1=p.canny_alto, param2=p.acumulador,
        minRadius=p.r_min, maxRadius=p.r_max,
    )
    if circ is None:
        return np.zeros((0, 3), np.float32)
    c = suprimir_sobrepostos(circ[0])
    if p.validar_contraste and len(c):
        c = np.array([k for k in c if _contraste(proc, *k) >= p.contraste_min],
                     dtype=np.float32).reshape(-1, 3)
    return c


def detectar_crateras(img: np.ndarray, p: ParamsDeteccao | None = None) -> np.ndarray:
    """Devolve as crateras como array float32 (N, 3) com colunas (x, y, r)."""
    p = p or ParamsDeteccao()
    proc = preprocessar(img, p)
    if p.metodo == "sombra":
        return _detectar_sombra(proc, p)
    if p.metodo == "hough":
        return _detectar_hough(proc, p)
    raise ValueError(f"método desconhecido: {p.metodo!r} (use 'sombra' ou 'hough')")
