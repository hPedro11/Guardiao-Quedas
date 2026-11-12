"""Aplicacao em tempo real. Uso: python -m src.app  (webcam)  |  python -m src.app --fonte video.mp4"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from .pipeline import Pipeline, desenhar
from .regras_queda import QUEDA

LOG = Path(__file__).resolve().parent.parent / "results" / "eventos.csv"


def registrar_evento(fonte):
    novo = not LOG.exists()
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["data_hora", "fonte", "evento"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), fonte, "QUEDA"])


def main():
    ap = argparse.ArgumentParser(description="Deteccao de queda pela webcam ou por video")
    ap.add_argument("--fonte", default="0", help="indice da webcam (0, 1...) ou caminho de video")
    ap.add_argument("--modelo", default="lite", choices=["lite", "full", "heavy"])
    ap.add_argument("--sem-preproc", action="store_true", help="desliga filtro e equalizacao")
    ap.add_argument("--mascara", action="store_true", help="mostra a mascara de movimento")
    args = ap.parse_args()

    webcam = args.fonte.isdigit()
    cap = cv2.VideoCapture(int(args.fonte) if webcam else args.fonte)
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir a fonte: {args.fonte}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    pipeline = Pipeline(args.modelo, preproc=not args.sem_preproc)

    inicio = anterior = time.perf_counter()
    n, estado_anterior, fps = 0, None, 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if webcam:
            frame = cv2.flip(frame, 1)
        agora = time.perf_counter()
        t = agora - inicio if webcam else n / fps_video
        r = pipeline.processar(frame, t)
        if r.estado == QUEDA and estado_anterior != QUEDA:
            registrar_evento(args.fonte)
            print(f"[ALERTA] queda detectada em t={t:.1f}s")
        estado_anterior = r.estado
        if agora > anterior:
            fps = 0.9 * fps + 0.1 / (agora - anterior) if fps else 1 / (agora - anterior)
        anterior = agora
        n += 1
        cv2.imshow("Guardiao de Quedas", desenhar(frame, r, fps))
        if args.mascara:
            cv2.imshow("Movimento (MOG2)", r.mascara)
        if cv2.waitKey(5) & 0xFF == 27:   # Esc
            break
    cap.release()
    cv2.destroyAllWindows()
    pipeline.fechar()


if __name__ == "__main__":
    main()
