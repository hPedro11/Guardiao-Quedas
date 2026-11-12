"""Deteccao de movimento: subtracao de fundo MOG2, morfologia e contornos."""
import cv2
import numpy as np


class AnalisadorMovimento:
    def __init__(self, area_min=1500):
        self.subtrator = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.area_min = area_min

    def analisar(self, frame_bgr):
        """Retorna (mascara, fracao de pixels em movimento, bbox do maior objeto ou None)."""
        fg = self.subtrator.apply(frame_bgr)
        # sombras saem em cinza (127): o limiar deixa so o primeiro plano
        _, mascara = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self.kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, self.kernel)
        fracao = float(np.count_nonzero(mascara)) / mascara.size
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(maior) >= self.area_min:
                bbox = cv2.boundingRect(maior)
        return mascara, fracao, bbox
