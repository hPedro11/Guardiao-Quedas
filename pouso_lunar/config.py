"""Parâmetros do projeto. Todo valor ajustável fica aqui, com a justificativa.

Os comentários servem de roteiro para a arguição: cada número tem um porquê.
"""

from pathlib import Path

# ----------------------------------------------------------------- caminhos
# Ancorados na raiz do projeto (e não no diretório atual) para funcionar
# tanto em `python main.py` quanto nos notebooks, que rodam dentro de notebooks/.
RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "dados"
PASTA_RESULTADOS = RAIZ / "resultados"

# ------------------------------------------------------------- reprodução
SEMENTE = 42  # toda aleatoriedade (amostras, trajetórias) usa esta semente

# --------------------------------------------------------------- constantes
RAIO_LUA_KM = 1737.4  # raio médio da Lua (IAU), o mesmo usado pelo mosaico

# ---------------------------------------------------------- fontes de dados
# Mosaico global LROC WAC (NASA/LRO, ASU), 100 m/px, projeção cilíndrica simples.
# BigTIFF sem compressão e com 1 linha por strip: dá para baixar só a região
# de estudo com requisições HTTP Range, sem baixar os 5,96 GB.
URL_MOSAICO = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
)
MOSAICO_LARGURA = 109164  # pixels (360°)
MOSAICO_ALTURA = 54582  # pixels (180°)
MOSAICO_OFFSET_TABELA_STRIPS = 437048  # byte onde começa StripOffsets (uint64 LE)
PX_POR_GRAU_NATIVO = MOSAICO_LARGURA / 360.0  # ≈ 303,23 px/grau ≈ 100 m/px

# Catálogo de crateras de Robbins (2018), USGS Astrogeology. ~1,3 milhão de
# crateras ≥ 1 km medidas sobre imagens LROC WAC: é o nosso gabarito.
URL_CATALOGO = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
    "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
    "lunar_crater_database_robbins_2018"
)

# --------------------------------------------------------- região de estudo
# Faixa equatorial (|lat| ≤ 30°) para a distorção da projeção ficar pequena.
# Esta janela pega a borda de Mare Tranquillitatis (mar liso, poucas crateras)
# com as terras altas ao sul (muitas crateras): testa os dois tipos de terreno.
LAT_MIN, LAT_MAX = -5.0, 15.0
LON_MIN, LON_MAX = 15.0, 35.0

# O recorte nativo tem ~6065 px de lado. Trabalhamos com metade (0,2 km/px):
# as crateras de interesse (≥ 2 km) continuam com ≥ 10 px e tudo fica 4x mais rápido.
ESCALA_MAPA = 0.5

# Lado do frame da câmera simulada (px). 480 é a resolução típica de uma
# câmera de navegação e mantém a demo em tempo real.
TAM_FRAME = 480

# ------------------------------------------------- localização
# ORB: rápido e com descritor binário (distância de Hamming). O mapa recebe
# muitos pontos porque é grande; o frame, poucos, para ficar em tempo real.
# (50000/3000 → 483 ms/frame; 30000/1500 → 171 ms/frame com o mesmo sucesso.)
ORB_FEATURES_MAPA = 30000
ORB_FEATURES_FRAME = 1500
# SIFT: sem limite de pontos (limitar derruba o sucesso de 0,72 para 0,45,
# porque sobram só os pontos mais fortes, concentrados). Com ~70 mil pontos no
# mapa, o casamento usa FLANN (KD-tree aproximada) em vez de força bruta:
# 415 ms/frame contra 1757 ms.
# Teste da razão de Lowe: só aceita o casamento se o melhor vizinho for
# claramente melhor que o segundo (d1 < 0,75·d2). Elimina casamentos ambíguos
# em terreno repetitivo (muitas crateras parecidas).
RAZAO_LOWE = 0.75
# RANSAC: um ponto é inlier se, depois de projetado pela homografia, cai a
# menos de 5 px de onde deveria. Abaixo de 15 inliers a estimativa não é confiável.
LIMIAR_RANSAC_PX = 5.0
MIN_INLIERS = 15
# Equalizar o histograma do mapa e do frame antes das features.
# O mosaico é escuro; sem isso, mudanças de gama/brilho derrubam o ORB.
EQUALIZAR_LOCALIZACAO = True
# Template matching (baseline): correlação normalizada mínima para aceitar.
LIMIAR_TEMPLATE = 0.5

# Direção sombra→brilho no mapa (graus; 0 = leste/+x). Em 97% das crateras
# do catálogo a metade oeste é mais escura: o sol do mosaico vem do oeste.
DIRECAO_SOL_GRAUS = 0.0

# Parâmetros do detector escolhidos pela ablação do notebook 02, olhando só a
# metade de ajuste do mapa (melhor F1 = 0,429 no ajuste; 0,473 no teste).
# Sobrescrevem os padrões de detector.ParamsDeteccao.
PARAMS_DETECCAO: dict = {
    "metodo": "sombra",
    "usar_clahe": False,  # CLAHE realça a textura e cria pares falsos (F1 0,31 com × 0,43 sem)
    "limiar_sombra": 18,  # 15 → mais recall, 21 → mais precisão; 18 equilibra (maior F1)
    "janela_media": 21,  # ~4 km: maior que as crateras pequenas, menor que as variações de albedo
    "r_min": 5,
    "r_max": 60,
}


# Menor diâmetro (px, no mapa de trabalho) que entra no gabarito. Abaixo disso
# a cratera ocupa poucos pixels e não é resolvida de forma confiável.
DIAM_MIN_PX = 10
# Maior diâmetro (px) avaliado = 2·r_max do detector (24 km a 0,2 km/px).
# O catálogo inclui bacias de centenas de km (a própria Tranquillitatis), que
# não cabem num frame da câmera e estão fora do escopo do detector.
DIAM_MAX_PX = 120
