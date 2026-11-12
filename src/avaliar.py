"""Avaliacao nos clipes gravados. Uso: python -m src.avaliar

Espera os videos em data/clips/queda/ e data/clips/nao_queda/.
Gera em results/: resultados.csv, metricas.json, matrizes de confusao, comparacao e curvas por clipe.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metricas import matriz_confusao, metricas
from .pipeline import Pipeline
from .regras_queda import QUEDA

RAIZ = Path(__file__).resolve().parent.parent
CLIPES = RAIZ / "data" / "clips"
RESULTADOS = RAIZ / "results"
EXTENSOES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CONFIGS = {"com_preproc": True, "sem_preproc": False}


def listar_clipes():
    """Retorna [(caminho, e_queda)] em ordem alfabetica."""
    clipes = []
    for pasta, e_queda in (("queda", True), ("nao_queda", False)):
        for arq in sorted((CLIPES / pasta).iterdir()):
            if arq.suffix.lower() in EXTENSOES:
                clipes.append((arq, e_queda))
    return clipes


def avaliar_clipe(caminho, modelo, preproc):
    cap = cv2.VideoCapture(str(caminho))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    pipeline = Pipeline(modelo, preproc=preproc)
    serie, n, com_pose = [], 0, 0
    inicio = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = n / fps_video
        r = pipeline.processar(frame, t)
        if r.medida is not None:
            com_pose += 1
            serie.append((t, r.medida.angulo_tronco, r.medida.quadril_y, r.estado))
        n += 1
    duracao = time.perf_counter() - inicio
    cap.release()
    eventos = pipeline.detector.eventos
    pipeline.fechar()
    return {
        "previsto": bool(eventos),
        "t_evento": round(eventos[0], 2) if eventos else None,
        "frames": n,
        "fps": round(n / duracao, 1) if duracao else 0.0,
        "taxa_pose": round(com_pose / n, 3) if n else 0.0,
        "serie": serie,
    }


def plotar_curva(serie, titulo, t_evento, destino):
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    if serie:
        ts = [s[0] for s in serie]
        ax1.plot(ts, [s[1] for s in serie], color="tab:red", label="angulo do tronco (graus)")
        ax1.set_ylim(0, 95)
        ax2 = ax1.twinx()
        ax2.plot(ts, [s[2] for s in serie], color="tab:blue", label="altura do quadril (y)")
        ax2.set_ylim(1, 0)
        ax2.set_ylabel("quadril y (0 = topo)")
    if t_evento is not None:
        ax1.axvline(t_evento, color="black", linestyle="--", label="queda confirmada")
    ax1.set_xlabel("tempo (s)")
    ax1.set_ylabel("angulo do tronco")
    ax1.set_title(titulo)
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def plotar_matriz(mc, titulo, destino):
    dados = [[mc["vp"], mc["fn"]], [mc["fp"], mc["vn"]]]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(dados, cmap="Blues")
    ax.set_xticks([0, 1], ["previsto: queda", "previsto: nao queda"])
    ax.set_yticks([0, 1], ["real: queda", "real: nao queda"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, dados[i][j], ha="center", va="center", fontsize=16)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def plotar_comparacao(resumo, destino):
    nomes = ["acuracia", "precisao", "recall", "f1", "especificidade"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    largura = 0.8 / len(resumo)
    for k, (config, dados) in enumerate(resumo.items()):
        xs = [i + k * largura for i in range(len(nomes))]
        ax.bar(xs, [dados["metricas"][n] for n in nomes], largura,
               label=f"{config} ({dados['fps_medio']:.0f} FPS)")
    ax.set_xticks([i + largura * (len(resumo) - 1) / 2 for i in range(len(nomes))], nomes)
    ax.set_ylim(0, 1.05)
    ax.set_title("Metricas por configuracao")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Avalia o detector nos clipes gravados")
    ap.add_argument("--modelo", default="lite", choices=["lite", "full", "heavy"])
    args = ap.parse_args()

    clipes = listar_clipes()
    if not clipes:
        raise SystemExit("Nenhum video em data/clips/queda ou data/clips/nao_queda.")
    RESULTADOS.mkdir(exist_ok=True)
    linhas, resumo = [], {}
    for config, preproc in CONFIGS.items():
        pasta_curvas = RESULTADOS / "curvas" / config
        pasta_curvas.mkdir(parents=True, exist_ok=True)
        reais, previstos, fps, taxas = [], [], [], []
        for caminho, e_queda in clipes:
            r = avaliar_clipe(caminho, args.modelo, preproc)
            print(f"[{config}] {caminho.name}: real={'queda' if e_queda else 'nao_queda'} "
                  f"previsto={'queda' if r['previsto'] else 'nao_queda'} ({r['fps']} FPS)")
            plotar_curva(r["serie"], f"{caminho.name} ({config})", r["t_evento"],
                         pasta_curvas / f"{caminho.parent.name}_{caminho.stem}.png")
            reais.append(e_queda)
            previstos.append(r["previsto"])
            fps.append(r["fps"])
            taxas.append(r["taxa_pose"])
            linhas.append([config, f"{caminho.parent.name}/{caminho.name}", e_queda, r["previsto"], r["t_evento"],
                           r["frames"], r["fps"], r["taxa_pose"]])
        mc = matriz_confusao(reais, previstos)
        resumo[config] = {
            "matriz": mc,
            "metricas": {k: round(v, 3) for k, v in metricas(mc).items()},
            "fps_medio": round(sum(fps) / len(fps), 1),
            "taxa_pose_media": round(sum(taxas) / len(taxas), 3),
        }
        plotar_matriz(mc, f"Matriz de confusao ({config})",
                      RESULTADOS / f"matriz_confusao_{config}.png")
    plotar_comparacao(resumo, RESULTADOS / "comparacao_metricas.png")

    with open(RESULTADOS / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "clipe", "real_queda", "previsto_queda", "t_evento_s",
                    "frames", "fps", "taxa_pose"])
        w.writerows(linhas)
    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": args.modelo,
        "n_clipes": len(clipes),
        "n_queda": sum(e for _, e in clipes),
        "configs": resumo,
    }
    (RESULTADOS / "metricas.json").write_text(json.dumps(saida, indent=2), encoding="utf-8")
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
