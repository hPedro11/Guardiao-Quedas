"""Regra de decisao de queda (maquina de estados). Nao depende de camera nem de modelo."""
import math
from collections import deque
from dataclasses import dataclass

OK, SUSPEITA, QUEDA = "OK", "SUSPEITA", "QUEDA"


@dataclass
class Medida:
    angulo_tronco: float      # graus em relacao a vertical (0 = em pe, 90 = deitado)
    quadril_y: float          # altura do quadril normalizada (0 = topo, 1 = base da imagem)
    movimento: float | None = None  # fracao de pixels em movimento (MOG2); None = nao medido


def angulo_tronco(ombro, quadril):
    """Angulo entre o segmento quadril->ombro e a vertical, com pontos em pixels."""
    dx = ombro[0] - quadril[0]
    dy = ombro[1] - quadril[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


class DetectorQueda:
    """OK -> SUSPEITA (quadril desce rapido) -> QUEDA (tronco fica na horizontal)."""

    def __init__(self, angulo_min=55.0, descida_min=0.15, janela_s=1.0,
                 confirmacao_s=1.0, espera_s=3.0, recuperacao_s=1.0,
                 folga_angulo=15.0, movimento_min=0.005):
        self.angulo_min = angulo_min          # tronco "deitado" a partir deste angulo
        self.descida_min = descida_min        # descida do quadril (fracao da altura) na janela
        self.janela_s = janela_s              # janela em que a descida e medida
        self.confirmacao_s = confirmacao_s    # tempo deitado para confirmar a queda
        self.espera_s = espera_s              # tempo maximo em SUSPEITA sem deitar
        self.recuperacao_s = recuperacao_s    # tempo em pe para sair de QUEDA
        self.folga_angulo = folga_angulo      # histerese para considerar "em pe" de novo
        self.movimento_min = movimento_min
        self.estado = OK
        self.eventos = []                     # instantes (s) em que uma queda foi confirmada
        self.descida = 0.0
        self._hist = deque()
        self._t_suspeita = self._t_deitado = self._t_em_pe = None

    def atualizar(self, t, medida=None):
        """Recebe o instante t (s) e a medida do quadro (None se a pose nao foi achada)."""
        deitado = None
        disparo = False
        self.descida = 0.0
        if medida is not None:
            self._hist.append((t, medida.quadril_y))
            while t - self._hist[0][0] > self.janela_s:
                self._hist.popleft()
            self.descida = medida.quadril_y - min(y for _, y in self._hist)
            deitado = medida.angulo_tronco >= self.angulo_min
            mov_ok = medida.movimento is None or medida.movimento >= self.movimento_min
            disparo = self.descida >= self.descida_min and mov_ok

        if self.estado == OK and disparo:
            self.estado, self._t_suspeita, self._t_deitado = SUSPEITA, t, None

        if self.estado == SUSPEITA:
            if deitado:
                if self._t_deitado is None:
                    self._t_deitado = t
                if t - self._t_deitado >= self.confirmacao_s:
                    self.estado, self._t_em_pe = QUEDA, None
                    self.eventos.append(t)
            elif deitado is False:
                self._t_deitado = None
            if (self.estado == SUSPEITA and self._t_deitado is None
                    and t - self._t_suspeita > self.espera_s):
                self.estado = OK
        elif self.estado == QUEDA and medida is not None:
            if medida.angulo_tronco < self.angulo_min - self.folga_angulo:
                if self._t_em_pe is None:
                    self._t_em_pe = t
                if t - self._t_em_pe >= self.recuperacao_s:
                    self.estado = OK
                    self._hist.clear()
            else:
                self._t_em_pe = None
        return self.estado
