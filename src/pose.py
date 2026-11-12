"""Estimativa de pose com MediaPipe Tasks API."""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

from .regras_queda import angulo_tronco

PASTA_MODELOS = Path(__file__).resolve().parent.parent / "models"
URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task")

OMBRO_E, OMBRO_D, QUADRIL_E, QUADRIL_D = 11, 12, 23, 24
CONEXOES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
            (26, 28), (11, 13), (13, 15), (12, 14), (14, 16)]


def caminho_modelo(nome="lite"):
    """Baixa o modelo (lite, full ou heavy) na primeira execucao."""
    destino = PASTA_MODELOS / f"pose_landmarker_{nome}.task"
    if not destino.exists():
        PASTA_MODELOS.mkdir(exist_ok=True)
        print(f"Baixando modelo {nome}...")
        urlretrieve(URL.format(nome), destino)
    return str(destino)


class EstimadorPose:
    def __init__(self, modelo="lite", confianca=0.5):
        opcoes = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=caminho_modelo(modelo)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=confianca)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opcoes)
        self._ultimo_ts = -1

    def detectar(self, frame_bgr, t):
        """Retorna a lista de 33 landmarks da pessoa, ou None se ninguem foi encontrado."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t * 1000), self._ultimo_ts + 1)   # o modo VIDEO exige tempo crescente
        self._ultimo_ts = ts
        resultado = self.landmarker.detect_for_video(mp_image, timestamp_ms=ts)
        return resultado.pose_landmarks[0] if resultado.pose_landmarks else None

    def fechar(self):
        self.landmarker.close()


def extrair_tronco(landmarks, largura, altura, visibilidade_min=0.5):
    """Retorna (angulo do tronco em graus, y do quadril normalizado) ou None."""
    pontos = [landmarks[i] for i in (OMBRO_E, OMBRO_D, QUADRIL_E, QUADRIL_D)]
    if any(p.visibility < visibilidade_min for p in pontos):
        return None
    oe, od, qe, qd = pontos
    ombro = ((oe.x + od.x) / 2 * largura, (oe.y + od.y) / 2 * altura)
    quadril = ((qe.x + qd.x) / 2 * largura, (qe.y + qd.y) / 2 * altura)
    return angulo_tronco(ombro, quadril), (qe.y + qd.y) / 2
