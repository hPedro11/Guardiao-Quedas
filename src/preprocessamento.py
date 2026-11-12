"""Pre-processamento do quadro: Labs 03 (equalizacao), 04 (filtro) e 05 (espaco de cor)."""
import cv2


def preprocessar(frame_bgr):
    """Suaviza o ruido e equaliza o brilho (canal V do HSV) para ambientes escuros."""
    suave = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    h, s, v = cv2.split(cv2.cvtColor(suave, cv2.COLOR_BGR2HSV))
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
