from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class ParamsDeteccao:
    metodo: str = "sombra"  # "sombra" ou "hough"

    # pré-processamento
    usar_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grade: int = 8
    blur_ksize: int = 3

    # tamanho das crateras que procuramos (raio em pixels)
    r_min: int = 5
    r_max: int = 60

    # método da sombra e brilho
    limiar_sombra: float = 18.0  # quanto mais escuro/claro que a vizinhança o pixel precisa ser
    janela_media: int = 21  # tamanho da vizinhança usada na média
    area_min: int = 4  # manchas menores que isso são ignoradas
    direcao_sol_graus: float = 0.0  # para onde aponta o caminho sombra -> brilho (0 = leste)
    ang_max: float = 40.0  # quanto o par pode fugir da direção do sol
    alcance_fator: float = 2.5  # a sombra e o brilho precisam estar perto um do outro
    razao_area_max: float = 6.0  # e ter tamanhos parecidos
    fator_raio: float = 0.8  # raio da cratera = 0.8 x distância entre as duas manchas

    # método HoughCircles
    dp: float = 1.2
    min_dist_fator: float = 0.8
    canny_alto: float = 100
    acumulador: float = 25  # quanto menor, mais círculos ele aceita


def preprocessar(img: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if p.usar_clahe:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(p.clahe_grade, p.clahe_grade))
        cinza = clahe.apply(cinza)
    if p.blur_ksize and p.blur_ksize > 1:
        k = p.blur_ksize | 1  # o tamanho do blur tem que ser ímpar
        cinza = cv2.GaussianBlur(cinza, (k, k), 0)
    return cinza


def suprimir_sobrepostos(c: np.ndarray) -> np.ndarray:
    # Quando dois círculos têm o centro quase no mesmo lugar, fica só o primeiro.
    # Para não comparar todos com todos (muito lento), dividimos a imagem em
    # quadradinhos e comparamos cada círculo só com os dos quadradinhos vizinhos.
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
    # centro e área de cada mancha branca da máscara
    _, _, stats, centros = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ok = areas >= area_min
    return centros[1:][ok].astype(np.float32), areas[ok].astype(np.float32)


def _detectar_sombra(proc: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    # Com o sol baixo, cada cratera aparece como uma mancha escura (sombra)
    # colada numa mancha clara (a outra parede, iluminada). Procuramos esses pares.
    f = proc.astype(np.float32)
    media = cv2.blur(f, (p.janela_media, p.janela_media))

    # limiar comparando com a vizinhança, porque o mar é escuro e as terras altas são claras
    sombra = np.uint8(f < media - p.limiar_sombra) * 255
    brilho = np.uint8(f > media + p.limiar_sombra) * 255

    # abertura para apagar pontinhos soltos
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
    for i in np.argsort(-area_s):  # começa pelas sombras maiores
        v = cb - cs[i]
        dist = np.hypot(v[:, 0], v[:, 1])
        ao_longo = v @ d
        transversal = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
        ang = np.degrees(np.arctan2(transversal, ao_longo))
        alcance = p.alcance_fator * math.sqrt(float(area_s[i])) + 3.0
        razao = np.maximum(area_b / area_s[i], area_s[i] / area_b)

        # a mancha clara tem que estar na direção do sol, perto e com tamanho parecido
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


def _detectar_hough(proc: np.ndarray, p: ParamsDeteccao) -> np.ndarray:
    circ = cv2.HoughCircles(
        proc, cv2.HOUGH_GRADIENT, dp=p.dp,
        minDist=max(1.0, p.min_dist_fator * p.r_min),
        param1=p.canny_alto, param2=p.acumulador,
        minRadius=p.r_min, maxRadius=p.r_max,
    )
    if circ is None:
        return np.zeros((0, 3), np.float32)
    return suprimir_sobrepostos(circ[0])


def detectar_crateras(img: np.ndarray, p: ParamsDeteccao | None = None) -> np.ndarray:
    # devolve uma lista de crateras, cada uma com (x, y, raio)
    p = p or ParamsDeteccao()
    proc = preprocessar(img, p)
    if p.metodo == "sombra":
        return _detectar_sombra(proc, p)
    if p.metodo == "hough":
        return _detectar_hough(proc, p)
    raise ValueError(f"método desconhecido: {p.metodo!r} (use 'sombra' ou 'hough')")
