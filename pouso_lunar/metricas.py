"""Métricas: casamento detecção × gabarito, precisão/recall/F1 e erro em km."""

import numpy as np
import pandas as pd


def casar_deteccoes(det, gab, tol_centro: float = 0.5, faixa_raio=(0.5, 2.0)) -> dict:
    """Casa crateras detectadas com o gabarito, um-para-um.

    Um par (detecção i, gabarito j) é candidato se a distância entre centros
    for < tol_centro·r_gab e a razão r_det/r_gab estiver em faixa_raio.
    Guloso: os candidatos são aceitos do mais próximo para o mais distante,
    e cada cratera (dos dois lados) só pode ser usada uma vez.
    """
    det = np.asarray(det, dtype=float).reshape(-1, 3)
    gab = np.asarray(gab, dtype=float).reshape(-1, 3)
    usado_det = np.zeros(len(det), bool)
    usado_gab = np.zeros(len(gab), bool)
    pares: list[tuple[int, int]] = []
    if len(det) and len(gab):
        dist = np.hypot(det[:, None, 0] - gab[None, :, 0], det[:, None, 1] - gab[None, :, 1])
        razao = det[:, None, 2] / gab[None, :, 2]
        cand = (dist < tol_centro * gab[None, :, 2]) & (razao >= faixa_raio[0]) & (razao <= faixa_raio[1])
        ii, jj = np.nonzero(cand)
        for k in np.argsort(dist[ii, jj], kind="stable"):
            i, j = int(ii[k]), int(jj[k])
            if not usado_det[i] and not usado_gab[j]:
                usado_det[i] = usado_gab[j] = True
                pares.append((i, j))
    tp = len(pares)
    return {
        "tp": tp,
        "fp": int(len(det) - tp),
        "fn": int(len(gab) - tp),
        "pares": pares,
        "mascara_det_tp": usado_det,
        "mascara_gab_tp": usado_gab,
    }


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precisão, recall e F1. Divisão por zero vale 0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def por_faixa(det, gab, faixas_px=((10, 20), (20, 40), (40, 1e9))) -> pd.DataFrame:
    """P/R/F1 por faixa de diâmetro (px).

    O gabarito entra na faixa pelo próprio diâmetro, e a detecção também.
    O casamento é feito uma vez com tudo; depois cada lado é contado na sua faixa.
    """
    det = np.asarray(det, dtype=float).reshape(-1, 3)
    gab = np.asarray(gab, dtype=float).reshape(-1, 3)
    m = casar_deteccoes(det, gab)
    linhas = []
    for lo, hi in faixas_px:
        nome = f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"
        g_in = (2 * gab[:, 2] >= lo) & (2 * gab[:, 2] < hi)
        d_in = (2 * det[:, 2] >= lo) & (2 * det[:, 2] < hi)
        tp = int((m["mascara_gab_tp"] & g_in).sum())
        fn = int(g_in.sum() - tp)
        fp = int((~m["mascara_det_tp"] & d_in).sum())
        p, r, f = prf1(tp, fp, fn)
        linhas.append({"faixa": nome, "tp": tp, "fp": fp, "fn": fn,
                       "precisao": p, "recall": r, "f1": f})
    return pd.DataFrame(linhas)


def erro_km(x_est, y_est, x_real, y_real, km_por_px: float):
    """Distância euclidiana em km entre a posição estimada e a real."""
    d = np.hypot(np.asarray(x_est, float) - x_real, np.asarray(y_est, float) - y_real) * km_por_px
    return float(d) if np.ndim(d) == 0 else d
