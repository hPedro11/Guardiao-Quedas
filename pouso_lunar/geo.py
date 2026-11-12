"""Conversões entre coordenadas lunares (lat/lon) e pixels do mapa.

O mosaico usa projeção cilíndrica simples (equiretangular): latitude e
longitude variam linearmente com a linha e a coluna. Por isso a conversão
é só uma escala (pixels por grau) e um deslocamento (canto superior esquerdo).
"""

from dataclasses import dataclass
import math

import numpy as np

from . import config


@dataclass(frozen=True)
class MapaInfo:
    """Georreferência de um recorte do mapa.

    lat_max/lon_min são as coordenadas do canto superior esquerdo (pixel 0,0).
    """

    lat_max: float
    lon_min: float
    px_por_grau: float
    largura: int
    altura: int

    @property
    def km_por_px(self) -> float:
        """Quilômetros por pixel (no equador; |lat| ≤ 30° mantém o erro pequeno)."""
        km_por_grau = 2 * math.pi * config.RAIO_LUA_KM / 360.0
        return km_por_grau / self.px_por_grau


def normalizar_lon(lon):
    """Leva a longitude para o intervalo [-180, 180). Ex.: 345° → -15°."""
    r = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    return float(r) if np.isscalar(lon) else r


def latlon_para_px(lat, lon, info: MapaInfo):
    """(lat, lon) em graus → (x, y) em pixels do mapa. Aceita escalares ou arrays."""
    x = (np.asarray(lon, dtype=float) - info.lon_min) * info.px_por_grau
    y = (info.lat_max - np.asarray(lat, dtype=float)) * info.px_por_grau
    if np.isscalar(lat) and np.isscalar(lon):
        return float(x), float(y)
    return x, y


def px_para_latlon(x, y, info: MapaInfo):
    """(x, y) em pixels → (lat, lon) em graus. Inversa de latlon_para_px."""
    lon = info.lon_min + np.asarray(x, dtype=float) / info.px_por_grau
    lat = info.lat_max - np.asarray(y, dtype=float) / info.px_por_grau
    if np.isscalar(x) and np.isscalar(y):
        return float(lat), float(lon)
    return lat, lon
