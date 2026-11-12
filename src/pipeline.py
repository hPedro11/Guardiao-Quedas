"""Junta pre-processamento, movimento, pose e a regra de queda em um unico passo por quadro."""
from dataclasses import dataclass

import cv2

from .movimento import AnalisadorMovimento
from .pose import CONEXOES, EstimadorPose, extrair_tronco
from .preprocessamento import preprocessar
from .regras_queda import OK, QUEDA, SUSPEITA, DetectorQueda, Medida

CORES = {OK: (0, 180, 0), SUSPEITA: (0, 200, 255), QUEDA: (0, 0, 255)}


@dataclass
class Resultado:
    estado: str
    medida: Medida | None
    landmarks: list | None
    bbox: tuple | None
    mascara: object


class Pipeline:
    def __init__(self, modelo="lite", preproc=True, detector=None):
        self.preproc = preproc
        self.movimento = AnalisadorMovimento()
        self.pose = EstimadorPose(modelo)
        self.detector = detector or DetectorQueda()

    def processar(self, frame_bgr, t):
        quadro = preprocessar(frame_bgr) if self.preproc else frame_bgr
        mascara, fracao, bbox = self.movimento.analisar(quadro)
        landmarks = self.pose.detectar(quadro, t)
        medida = None
        if landmarks is not None:
            altura, largura = frame_bgr.shape[:2]
            tronco = extrair_tronco(landmarks, largura, altura)
            if tronco is not None:
                medida = Medida(tronco[0], tronco[1], fracao)
        estado = self.detector.atualizar(t, medida)
        return Resultado(estado, medida, landmarks, bbox, mascara)

    def fechar(self):
        self.pose.fechar()


def desenhar(frame, r, fps=None):
    """Desenha esqueleto, caixa de movimento e a faixa de estado sobre o quadro."""
    altura, largura = frame.shape[:2]
    cor = CORES[r.estado]
    if r.bbox is not None:
        x, y, w, h = r.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)
    if r.landmarks is not None:
        pts = [(int(p.x * largura), int(p.y * altura)) for p in r.landmarks]
        for a, b in CONEXOES:
            cv2.line(frame, pts[a], pts[b], cor, 2)
        for i in {i for par in CONEXOES for i in par}:
            cv2.circle(frame, pts[i], 4, (255, 255, 255), -1)
    cv2.rectangle(frame, (0, 0), (largura, 40), cor, -1)
    texto = "QUEDA DETECTADA - ALERTA" if r.estado == QUEDA else r.estado
    cv2.putText(frame, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    linhas = []
    if r.medida is not None:
        linhas.append(f"tronco: {r.medida.angulo_tronco:.0f} graus")
        linhas.append(f"movimento: {r.medida.movimento * 100:.1f}%")
    if fps is not None:
        linhas.append(f"FPS: {fps:.0f}")
    for i, linha in enumerate(linhas):
        cv2.putText(frame, linha, (10, 65 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    return frame
