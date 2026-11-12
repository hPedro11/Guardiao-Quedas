"""Demo: descida simulada de uma sonda lunar com detecção de crateras e localização.

Uso:
    python main.py [--metodo orb|sift] [--semente N]

Teclas:
    espaço  pausa / continua
    d       liga / desliga a detecção de crateras
    m       alterna o método de localização (ORB <-> SIFT)
    s       salva um print em resultados/demo.png
    q, Esc  sai

Mouse: clique no minimapa (direita) para escolher um novo ponto de pouso.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from pouso_lunar import config
from pouso_lunar.dados import carregar_mapa
from pouso_lunar.detector import ParamsDeteccao, detectar_crateras
from pouso_lunar.localizador import Localizador
from pouso_lunar.metricas import erro_km
from pouso_lunar.simulador import centro_frame, direcao_no_frame, gerar_frame, trajetoria_descida
from pouso_lunar.visual import VERMELHO, Minimapa, desenhar_crateras, montar_painel

JANELA = "Pouso Lunar"


class Demo:
    def __init__(self, mapa, info, metodo: str, semente: int):
        self.mapa, self.info = mapa, info
        self.tam = config.TAM_FRAME
        self.rng = np.random.default_rng(semente)
        self.mini = Minimapa(mapa, self.tam)
        self.metodo = metodo
        self.loc = self._criar_localizador(metodo)
        self.detectar = True
        self.pausado = False
        self.novo_alvo = None
        self._nova_descida()

    def _criar_localizador(self, metodo: str) -> Localizador:
        print(f"Preparando localizador {metodo.upper()} (a 1a vez calcula e guarda em cache)...")
        cache = config.PASTA_DADOS / f"cache_{metodo}_{self.mapa.shape[1]}.npz"
        return Localizador(self.mapa, metodo, cache=cache)

    def _ponto_aleatorio(self):
        alt, larg = self.mapa.shape
        return float(self.rng.uniform(0.15, 0.85) * larg), float(self.rng.uniform(0.15, 0.85) * alt)

    def _nova_descida(self, inicio=None, alvo=None, zoom_ini: float = 0.5):
        inicio = inicio or self._ponto_aleatorio()
        self.alvo = alvo or self._ponto_aleatorio()
        alt, larg = self.mapa.shape
        n = max(60, int(240 * (1.6 - zoom_ini) / 1.1))
        self.traj = trajetoria_descida(inicio, self.alvo, n_frames=n, zoom_ini=zoom_ini,
                                       giro_total_graus=float(self.rng.uniform(-60, 60)),
                                       largura=larg, altura=alt, tam=self.tam)
        # Um alvo clicado perto da borda é preso à área onde a câmera cabe:
        # o marcador mostra onde a sonda realmente vai pousar.
        self.alvo = self.traj[-1][:2]
        self.i = 0
        self.real, self.est = [], []
        self.ultimo_ok = None

    def ao_clicar(self, evento, u, v, flags, _):
        if evento == cv2.EVENT_LBUTTONDOWN and u >= self.tam:
            self.novo_alvo = self.mini.para_mapa(u - self.tam, v)

    def _localizar(self, frame, zoom):
        if self.ultimo_ok is not None:
            # Rastreamento: procura só perto da última posição conhecida (mais rápido).
            lx, ly = self.ultimo_ok
            m = 1.5 * self.tam / zoom
            r = self.loc.localizar(frame, janela=(int(lx - m), int(ly - m), int(lx + m), int(ly + m)))
            if r.ok:
                return r
        return self.loc.localizar(frame)  # perdido: busca no mapa inteiro

    def passo(self):
        if self.novo_alvo is not None:
            x, y, _, zoom = self.traj[min(self.i, len(self.traj) - 1)]
            self._nova_descida(inicio=(x, y), alvo=self.novo_alvo, zoom_ini=min(zoom, 1.4))
            self.novo_alvo = None
        if self.i >= len(self.traj):
            self._nova_descida()

        x, y, rumo, zoom = self.traj[self.i]
        frame, verdade = gerar_frame(self.mapa, x, y, rumo, zoom, tam=self.tam, ruido_sigma=3.0, rng=self.rng)
        t0 = time.perf_counter()
        r = self._localizar(frame, zoom)
        crateras = np.zeros((0, 3))
        if self.detectar:
            # Os raios crescem com o zoom da câmera, e a direção do sol na
            # imagem gira junto com a sonda (atitude conhecida pela IMU).
            p = ParamsDeteccao(**config.PARAMS_DETECCAO)
            pz = ParamsDeteccao(**{**p.__dict__,
                                   "r_min": max(3, round(p.r_min * zoom)),
                                   "r_max": max(6, round(p.r_max * zoom)),
                                   "direcao_sol_graus": direcao_no_frame(verdade.M, config.DIRECAO_SOL_GRAUS)})
            crateras = detectar_crateras(frame, pz)
        fps = 1.0 / max(1e-6, time.perf_counter() - t0)

        self.real.append((x, y))
        if r.ok:
            self.ultimo_ok = (r.x, r.y)
            self.est.append((r.x, r.y))
            erro = f"erro: {erro_km(r.x, r.y, x, y, self.info.km_por_px):.2f} km"
            status, cor_status = "LOCALIZADO", (0, 220, 0)
        else:
            self.ultimo_ok = None
            erro, status, cor_status = "erro: --", "PERDIDO", VERMELHO

        quadro = None
        if r.ok:
            c = self.tam - 1
            quadro = cv2.perspectiveTransform(
                np.float32([[[0, 0], [c, 0], [c, c], [0, c]]]), r.H)[0]
        img_frame = desenhar_crateras(frame, crateras) if self.detectar else frame
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR) if img_frame.ndim == 2 else img_frame
        cv2.drawMarker(img_frame, (int(centro_frame(self.tam)),) * 2, (255, 255, 0),
                       cv2.MARKER_CROSS, 16, 1)
        textos = [status, f"metodo: {self.metodo.upper()}", f"inliers: {r.n_inliers}", erro,
                  f"crateras: {len(crateras) if self.detectar else '-'}", f"FPS: {fps:.0f}",
                  f"altitude rel.: {1 / zoom:.2f}"]
        painel = montar_painel(img_frame, self.mini.desenhar(self.real, self.est, self.alvo, quadro),
                               textos, [cor_status])
        self.i += 1
        return painel

    def rodar(self):
        cv2.namedWindow(JANELA, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(JANELA, self.ao_clicar)
        painel = self.passo()
        while True:
            cv2.imshow(JANELA, painel)
            tecla = cv2.waitKey(1 if not self.pausado else 50) & 0xFF
            if tecla in (ord("q"), 27) or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord(" "):
                self.pausado = not self.pausado
            elif tecla == ord("d"):
                self.detectar = not self.detectar
            elif tecla == ord("m"):
                self.metodo = "sift" if self.metodo == "orb" else "orb"
                self.loc = self._criar_localizador(self.metodo)
                self.ultimo_ok = None
            elif tecla == ord("s"):
                config.PASTA_RESULTADOS.mkdir(exist_ok=True)
                destino = config.PASTA_RESULTADOS / "demo.png"
                cv2.imwrite(str(destino), painel)
                print(f"print salvo em {destino}")
            if not self.pausado or self.novo_alvo is not None:
                painel = self.passo()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Demo de navegação de pouso lunar")
    ap.add_argument("--metodo", choices=["orb", "sift"], default="orb")
    ap.add_argument("--semente", type=int, default=config.SEMENTE)
    args = ap.parse_args(argv)
    try:
        mapa, info = carregar_mapa()
    except FileNotFoundError as e:
        print(e)
        return 1
    Demo(mapa, info, args.metodo, args.semente).rodar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
