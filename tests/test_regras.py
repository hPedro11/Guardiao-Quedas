import numpy as np

from src.metricas import matriz_confusao, metricas
from src.movimento import AnalisadorMovimento
from src.preprocessamento import preprocessar
from src.regras_queda import OK, QUEDA, SUSPEITA, DetectorQueda, Medida, angulo_tronco

FPS = 15


def simular(trechos, detector=None):
    """trechos: [(duracao_s, angulo_ini, angulo_fim, y_ini, y_fim)]. Retorna (detector, estados)."""
    d = detector or DetectorQueda()
    estados, t = [], 0.0
    for dur, a0, a1, y0, y1 in trechos:
        n = int(dur * FPS)
        for i in range(n):
            k = i / max(n - 1, 1)
            m = None if a0 is None else Medida(a0 + (a1 - a0) * k, y0 + (y1 - y0) * k, 0.05)
            estados.append(d.atualizar(t, m))
            t += 1 / FPS
    return d, estados


def test_angulo_tronco():
    assert angulo_tronco((100, 100), (100, 200)) == 0
    assert angulo_tronco((200, 200), (100, 200)) == 90
    assert round(angulo_tronco((200, 100), (100, 200))) == 45


def test_queda_e_detectada_uma_vez():
    d, estados = simular([(2, 5, 5, 0.5, 0.5), (0.6, 5, 85, 0.5, 0.85), (3, 85, 85, 0.85, 0.85)])
    assert estados[-1] == QUEDA
    assert len(d.eventos) == 1
    assert 3.2 <= d.eventos[0] <= 4.0


def test_em_pe_nao_alerta():
    d, estados = simular([(5, 5, 8, 0.5, 0.5)])
    assert set(estados) == {OK} and d.eventos == []


def test_sentar_rapido_nao_alerta():
    d, estados = simular([(2, 5, 5, 0.5, 0.5), (0.6, 5, 15, 0.5, 0.72), (5, 15, 15, 0.72, 0.72)])
    assert SUSPEITA in estados
    assert estados[-1] == OK and d.eventos == []


def test_deitar_devagar_nao_alerta():
    d, estados = simular([(2, 5, 5, 0.5, 0.5), (6, 5, 85, 0.5, 0.85), (3, 85, 85, 0.85, 0.85)])
    assert d.eventos == []


def test_abaixar_para_pegar_objeto_nao_alerta():
    d, estados = simular([(2, 5, 5, 0.5, 0.5), (0.8, 5, 75, 0.5, 0.55), (1, 75, 5, 0.55, 0.5)])
    assert d.eventos == []


def test_levantar_volta_para_ok():
    d, estados = simular([(1, 5, 5, 0.5, 0.5), (0.6, 5, 85, 0.5, 0.85), (2, 85, 85, 0.85, 0.85),
                          (2, 85, 5, 0.85, 0.5), (2, 5, 5, 0.5, 0.5)])
    assert QUEDA in estados and estados[-1] == OK and len(d.eventos) == 1


def test_pose_perdida_depois_da_queda_mantem_alerta():
    d, estados = simular([(1, 5, 5, 0.5, 0.5), (0.6, 5, 85, 0.5, 0.85), (2, 85, 85, 0.85, 0.85),
                          (3, None, None, None, None)])
    assert estados[-1] == QUEDA


def test_pose_perdida_sem_queda_nao_trava_em_suspeita():
    d, estados = simular([(1, 5, 5, 0.5, 0.5), (0.6, 5, 15, 0.5, 0.72), (5, None, None, None, None)])
    assert estados[-1] == OK and d.eventos == []


def test_sem_movimento_nao_dispara():
    d = DetectorQueda()
    t = 0.0
    for y in (0.5, 0.5, 0.9, 0.9):   # salto do quadril com imagem parada = erro da pose
        d.atualizar(t, Medida(85, y, 0.0))
        t += 0.2
    for _ in range(20):
        d.atualizar(t, Medida(85, 0.9, 0.0))
        t += 0.2
    assert d.eventos == []


def test_metricas():
    mc = matriz_confusao([True, True, True, False, False], [True, True, False, True, False])
    assert mc == {"vp": 2, "fp": 1, "vn": 1, "fn": 1}
    m = metricas(mc)
    assert m["acuracia"] == 0.6
    assert round(m["precisao"], 3) == 0.667 and round(m["recall"], 3) == 0.667
    assert m["especificidade"] == 0.5
    assert metricas({"vp": 0, "fp": 0, "vn": 0, "fn": 0})["f1"] == 0.0


def test_movimento_encontra_objeto_que_se_move():
    a = AnalisadorMovimento(area_min=500)
    fundo = np.full((240, 320, 3), 40, np.uint8)
    for _ in range(30):
        a.analisar(fundo)
    quadro = fundo.copy()
    quadro[100:180, 50:90] = 220
    mascara, fracao, bbox = a.analisar(quadro)
    assert fracao > 0.02 and bbox is not None
    x, y, w, h = bbox
    assert abs(x - 50) <= 6 and abs(y - 100) <= 6 and h > w
    assert mascara.shape == (240, 320)


def test_preprocessar_mantem_formato():
    quadro = np.random.default_rng(0).integers(0, 60, (120, 160, 3), dtype=np.uint8)
    saida = preprocessar(quadro)
    assert saida.shape == quadro.shape and saida.dtype == np.uint8
    assert saida.max() > quadro.max()   # imagem escura fica mais clara
