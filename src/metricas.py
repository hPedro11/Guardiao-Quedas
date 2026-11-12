"""Metricas de classificacao binaria (queda = positivo), calculadas na mao."""


def matriz_confusao(reais, previstos):
    """reais/previstos: listas de bool. Retorna dict com vp, fp, vn, fn."""
    pares = list(zip(reais, previstos))
    return {
        "vp": sum(r and p for r, p in pares),
        "fp": sum((not r) and p for r, p in pares),
        "vn": sum((not r) and (not p) for r, p in pares),
        "fn": sum(r and (not p) for r, p in pares),
    }


def metricas(mc):
    def div(a, b):
        return a / b if b else 0.0
    total = mc["vp"] + mc["fp"] + mc["vn"] + mc["fn"]
    precisao = div(mc["vp"], mc["vp"] + mc["fp"])
    recall = div(mc["vp"], mc["vp"] + mc["fn"])
    return {
        "acuracia": div(mc["vp"] + mc["vn"], total),
        "precisao": precisao,
        "recall": recall,
        "f1": div(2 * precisao * recall, precisao + recall),
        "especificidade": div(mc["vn"], mc["vn"] + mc["fp"]),
    }
