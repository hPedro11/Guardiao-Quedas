from dataclasses import dataclass
import math

import numpy as np

from . import config


@dataclass(frozen=True)
class MapaInfo:
    # lat_max e lon_min são as coordenadas do canto de cima à esquerda do mapa
    lat_max: float
    lon_min: float
    px_por_grau: float
    largura: int
    altura: int

    @property
    def km_por_px(self) -> float:
        km_por_grau = 2 * math.pi * config.RAIO_LUA_KM / 360.0
        return km_por_grau / self.px_por_grau


def normalizar_lon(lon):
    # deixa a longitude entre -180 e 180 (ex.: 345 vira -15)
    r = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    return float(r) if np.isscalar(lon) else r


def latlon_para_px(lat, lon, info: MapaInfo):
    # No mapa, latitude e longitude viram linha e coluna de forma direta,
    # então basta multiplicar pelos pixels por grau.
    x = (np.asarray(lon, dtype=float) - info.lon_min) * info.px_por_grau
    y = (info.lat_max - np.asarray(lat, dtype=float)) * info.px_por_grau
    if np.isscalar(lat) and np.isscalar(lon):
        return float(x), float(y)
    return x, y
