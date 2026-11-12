from pathlib import Path

# pastas do projeto (a partir da raiz, para funcionar também dentro de notebooks/)
RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "dados"
PASTA_RESULTADOS = RAIZ / "resultados"

# semente fixa para os sorteios darem sempre o mesmo resultado
SEMENTE = 42

RAIO_LUA_KM = 1737.4

# Mosaico da Lua (LROC WAC, NASA), 100 metros por pixel
URL_MOSAICO = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
)
MOSAICO_LARGURA = 109164  # pixels (360 graus)
MOSAICO_ALTURA = 54582  # pixels (180 graus)
MOSAICO_OFFSET_TABELA_STRIPS = 437048  # posição, dentro do arquivo, da tabela de linhas
PX_POR_GRAU_NATIVO = MOSAICO_LARGURA / 360.0

# Catálogo de crateras de Robbins (USGS), usado para conferir as detecções
URL_CATALOGO = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
    "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
    "lunar_crater_database_robbins_2018"
)

# Região estudada: borda do Mare Tranquillitatis. Tem mar (liso, poucas
# crateras) e terras altas (muitas crateras), então dá para testar os dois.
LAT_MIN, LAT_MAX = -5.0, 15.0
LON_MIN, LON_MAX = 15.0, 35.0

# usamos o mapa na metade do tamanho (200 m por pixel) para ficar mais rápido
ESCALA_MAPA = 0.5

# tamanho da imagem da câmera simulada
TAM_FRAME = 480

# Localização
ORB_FEATURES_MAPA = 30000  # pontos ORB no mapa inteiro
ORB_FEATURES_FRAME = 1500  # pontos ORB em cada imagem da câmera
RAZAO_LOWE = 0.75  # teste de Lowe: o melhor par tem que ser bem melhor que o segundo
LIMIAR_RANSAC_PX = 5.0  # distância máxima (px) para o RANSAC aceitar um par
MIN_INLIERS = 15  # com menos pares confirmados que isso, consideramos que não achou
EQUALIZAR_LOCALIZACAO = True  # o mapa é escuro; equalizar ajudou muito o ORB
LIMIAR_TEMPLATE = 0.5  # nota mínima do template matching para aceitar

# O sol do mosaico vem do oeste (medimos no notebook 01), então a sombra fica
# a oeste e o brilho a leste. 0 graus = direção leste.
DIRECAO_SOL_GRAUS = 0.0

# Valores escolhidos para o detector no notebook 02
PARAMS_DETECCAO: dict = {
    "metodo": "sombra",
    "usar_clahe": False,  # com CLAHE apareciam mais crateras falsas
    "limiar_sombra": 18,  # 12 marcava muita coisa errada, 24 perdia muitas crateras
    "janela_media": 21,
    "r_min": 5,
    "r_max": 60,
}

# Tamanho das crateras do catálogo que usamos para conferir (em pixels).
# Menores que 10 px quase não aparecem na imagem; o catálogo também tem
# bacias enormes (centenas de km) que não aparecem como círculo.
DIAM_MIN_PX = 10
DIAM_MAX_PX = 120
