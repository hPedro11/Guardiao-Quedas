import numpy as np


def casar_deteccoes(det, gab, tol_centro: float = 0.5, faixa_raio=(0.5, 2.0)) -> dict:
    # Compara as crateras detectadas com as do catálogo. Conta como acerto quando
    # o centro está perto (menos de meio raio) e o tamanho é parecido (entre a
    # metade e o dobro). Cada cratera só pode ser usada uma vez, e os pares mais
    # próximos são casados primeiro.
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
        "tp": tp,  # acertos
        "fp": int(len(det) - tp),  # marcações que não são cratera do catálogo
        "fn": int(len(gab) - tp),  # crateras do catálogo que não foram achadas
        "pares": pares,
        "mascara_det_tp": usado_det,
        "mascara_gab_tp": usado_gab,
    }


def erro_km(x_est, y_est, x_real, y_real, km_por_px: float):
    # distância em km entre a posição encontrada e a verdadeira
    d = np.hypot(np.asarray(x_est, float) - x_real, np.asarray(y_est, float) - y_real) * km_por_px
    return float(d) if np.ndim(d) == 0 else d
