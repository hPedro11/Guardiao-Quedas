"""Gera docs/relatorio.pdf. Uso: python -m src.gerar_relatorio

Le integrantes.txt e, se existir, results/metricas.json (gerado por src.avaliar).
Rode de novo depois da avaliacao para o PDF sair com os numeros e graficos reais.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"
E = getSampleStyleSheet()


def p(texto):
    return Paragraph(texto, E["BodyText"])


def h(texto):
    return Paragraph(texto, E["Heading2"])


def tabela(linhas, larguras=None):
    celula = ParagraphStyle("celula", parent=E["BodyText"], fontSize=9, leading=11)
    linhas = [linhas[0]] + [[Paragraph(str(c), celula) for c in l] for l in linhas[1:]]
    t = Table(linhas, colWidths=larguras, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ED145B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def figura(caminho, largura=13 * cm):
    img = Image(str(caminho))
    img.drawHeight = largura * img.imageHeight / img.imageWidth
    img.drawWidth = largura
    return img


def integrantes():
    arq = RAIZ / "integrantes.txt"
    linhas = [l.strip() for l in arq.read_text(encoding="utf-8").splitlines()]
    return [l for l in linhas if l and not l.startswith("#")]


def secao_resultados():
    arq = RESULTADOS / "metricas.json"
    if not arq.exists():
        return [p("<b>RESULTADOS PENDENTES.</b> Grave os clipes em data/clips, rode "
                  "<i>python -m src.avaliar</i> e depois <i>python -m src.gerar_relatorio</i> "
                  "para preencher esta seção com as métricas e os gráficos reais.")]
    d = json.loads(arq.read_text(encoding="utf-8"))
    itens = [p(f"Avaliação gerada em {d['gerado_em']} com o modelo pose_landmarker_{d['modelo']}, "
               f"sobre {d['n_clipes']} clipes ({d['n_queda']} com queda e "
               f"{d['n_clipes'] - d['n_queda']} sem queda). Cada clipe conta como um caso: "
               "o sistema acerta um clipe de queda se confirmar pelo menos um evento QUEDA, "
               "e acerta um clipe sem queda se não confirmar nenhum."), Spacer(1, 6)]
    linhas = [["Configuração", "VP", "FP", "VN", "FN", "Acurácia", "Precisão", "Recall", "F1",
               "FPS", "Pose"]]
    for nome, c in d["configs"].items():
        m, mc = c["metricas"], c["matriz"]
        linhas.append([nome, mc["vp"], mc["fp"], mc["vn"], mc["fn"], f"{m['acuracia']:.2f}",
                       f"{m['precisao']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
                       f"{c['fps_medio']:.0f}", f"{c['taxa_pose_media'] * 100:.0f}%"])
    itens += [tabela(linhas, [3 * cm] + [1.1 * cm] * 4 + [1.7 * cm] * 4 + [1.2 * cm, 1.4 * cm]), Spacer(1, 6),
              p("VP = queda detectada; FN = queda perdida; FP = alarme falso; VN = atividade "
                "normal sem alarme. FPS é a velocidade média de processamento e Pose é a fração "
                "de quadros em que o tronco foi localizado."), Spacer(1, 8)]
    for nome in ["comparacao_metricas.png"] + [f"matriz_confusao_{c}.png" for c in d["configs"]]:
        if (RESULTADOS / nome).exists():
            itens += [figura(RESULTADOS / nome, 11 * cm), Spacer(1, 6)]
    curvas = sorted((RESULTADOS / "curvas").glob("*/*.png"))[:2]
    if curvas:
        itens.append(p("Exemplos de curvas por clipe (ângulo do tronco e altura do quadril ao "
                       "longo do tempo; a linha tracejada marca a queda confirmada):"))
        itens += [figura(c, 12 * cm) for c in curvas]
    itens += [Spacer(1, 6), p("<b>Análise crítica (preencher pela equipe):</b> comente quais "
              "clipes foram classificados errado e por que (ver resultados.csv e as curvas), "
              "e se o pré-processamento ajudou ou atrapalhou.")]
    return itens


def main():
    doc = SimpleDocTemplate(str(RAIZ / "docs" / "relatorio.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title="Guardião de Quedas - Relatório")
    c = [Paragraph("Guardião de Quedas", E["Title"]),
         p("Detecção de queda de idosos por visão computacional<br/>"
           "Global Solution 2026.2 (CitizenTech) - Cognitive Computing, Computer Vision and "
           "IoT Systems - Prof. Arnaldo Jr."),
         p("<b>Integrantes:</b> " + "; ".join(integrantes())), Spacer(1, 10)]

    c += [h("1. Contexto"),
          p("O tema CitizenTech pede tecnologia criada a partir de necessidades reais das "
            "pessoas. O enunciado cita, entre as inspirações, tecnologias de apoio a idosos "
            "para ampliar autonomia e segurança. Segundo a ficha informativa da Organização "
            "Mundial da Saúde sobre quedas (2021), as quedas são a segunda maior causa de morte "
            "por lesão não intencional no mundo, com cerca de 684 mil mortes por ano, e a "
            "faixa acima de 60 anos concentra o maior número de quedas fatais. Para quem mora "
            "sozinho, o tempo até alguém perceber a queda é parte do problema."),
          p("<b>Cidadão beneficiado:</b> idosos que moram sozinhos ou passam parte do dia sem "
            "acompanhante, e seus familiares e cuidadores. <b>ODS:</b> 3 (saúde e bem-estar), "
            "10 (redução das desigualdades) e 11 (cidades e comunidades sustentáveis)."),
          h("2. Problema e proposta"),
          p("Detectar automaticamente, com uma câmera comum (webcam), o momento em que uma "
            "pessoa cai e permanece no chão, e emitir um alerta. A solução roda localmente, "
            "sem enviar imagens para fora do computador e sem exigir que o idoso use pulseira "
            "ou botão de pânico."),
          h("3. Métodos")]
    c += [tabela([
        ["Etapa", "Técnica", "Lab"],
        ["Pre-processamento", "Filtro gaussiano 5x5 e equalização de histograma do canal V (HSV)",
         "03, 04, 05"],
        ["Movimento", "Subtração de fundo MOG2, limiar para remover sombras, abertura e "
         "fechamento morfológicos, maior contorno e boundingRect", "05, 06, 07"],
        ["Pose", "MediaPipe PoseLandmarker (Tasks API, modo VIDEO): ombros e quadris", "13"],
        ["Decisão", "Máquina de estados OK -> SUSPEITA -> QUEDA", "-"],
    ], [3.6 * cm, 10.9 * cm, 2.2 * cm]), Spacer(1, 8),
          p("<b>Medidas por quadro.</b> A partir do ponto médio dos ombros e do ponto médio "
            "dos quadris calculamos o ângulo do tronco em relação à vertical (0 graus em pé, "
            "90 graus deitado) e a altura do quadril na imagem (0 no topo, 1 na base). "
            "A máscara do MOG2 fornece a fração de pixels em movimento."),
          p("<b>Regra de decisão.</b> (1) SUSPEITA: o quadril desce pelo menos 15% da altura "
            "da imagem em até 1 segundo, com movimento na cena de pelo menos 0,5% dos pixels. "
            "(2) QUEDA: durante a suspeita, o tronco fica com 55 graus ou mais por 1 segundo "
            "seguido. (3) Se em 3 segundos o tronco não deitar, a suspeita é descartada. "
            "(4) O alerta só termina quando o tronco volta a menos de 40 graus por 1 segundo."),
          p("<b>Por que duas condições.</b> Só o ângulo confundiria deitar na cama ou abaixar "
            "para pegar um objeto com queda. Só a velocidade confundiria sentar rápido com "
            "queda. A combinação exige descida rápida seguida de permanência na horizontal. "
            "A exigência de movimento evita disparos quando a imagem está parada e a pose "
            "oscila sozinha."),
          h("4. Protocolo de avaliação"),
          p("Clipes curtos gravados pela equipe, separados em duas pastas: quedas simuladas "
            "(de lado, para frente, para trás) e atividades normais escolhidas para enganar o "
            "sistema (sentar rápido, agachar, abaixar para pegar um objeto, deitar devagar, "
            "andar). O script src/avaliar.py processa todos os clipes com e sem "
            "pré-processamento e calcula matriz de confusão, acurácia, precisão, recall, F1, "
            "FPS e taxa de quadros com pose encontrada."),
          h("5. Resultados")]
    c += secao_resultados()
    c += [h("6. Limitações"),
          p("- Quedas na direção da câmera mudam pouco o ângulo do tronco em 2D; a câmera "
            "deve ficar de lado para a área monitorada.<br/>"
            "- A descida do quadril é medida em fração da imagem, então depende da distância "
            "da pessoa à câmera.<br/>"
            "- O sistema acompanha uma pessoa por vez e depende de ombros e quadris visíveis; "
            "moveis na frente do corpo ou pouca luz reduzem a detecção.<br/>"
            "- As quedas do conjunto de teste são simuladas por adultos jovens, não quedas "
            "reais de idosos, e o conjunto é pequeno.<br/>"
            "- Os limiares foram definidos por regra, sem aprendizado a partir de dados."),
          h("7. Considerações finais"),
          p("A solução mostra que técnicas vistas na disciplina, combinadas, resolvem um "
            "problema concreto com hardware barato. Como próximos passos: enviar o alerta a um "
            "familiar (mensagem ou aplicativo), ajustar os limiares com mais clipes, suportar "
            "varias pessoas e avaliar com mais ângulos de câmera."),
          h("Referências"),
          p("Organização Mundial da Saúde. Falls (fact sheet), 26 abr. 2021. "
            "https://www.who.int/news-room/fact-sheets/detail/falls<br/>"
            "Material da disciplina: https://arnaldojr.github.io/cognitivecomputing/<br/>"
            "MediaPipe Pose Landmarker: "
            "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker")]
    doc.build(c)
    print("Relatório gerado em docs/relatorio.pdf")


if __name__ == "__main__":
    main()
