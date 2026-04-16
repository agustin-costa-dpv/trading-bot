"""
bot/xg_data.py
Fetchea datos de Expected Goals (xG) para enriquecer el análisis deportivo.

Fuente: football-data.co.uk — CSVs gratuitos con xG de las principales ligas.
Para ligas no cubiertas, devuelve None y el análisis sigue sin xG.

Uso:
    from bot.xg_data import get_xg_equipo, enriquecer_mercado_con_xg
"""

import logging
import unicodedata
import re
import io
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# URLs de CSVs por liga (football-data.co.uk)
# Temporada 2024-25
# ─────────────────────────────────────────────────────────────

LIGA_URLS = {
    # Liga: (url_csv, col_home, col_away, col_xg_home, col_xg_away)
    "Premier League":      "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "Championship":        "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "La Liga":             "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Bundesliga":          "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Bundesliga 2":        "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
    "Serie A":             "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Ligue 1":             "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "Eredivisie":          "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "Liga Portugal":       "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
}

# Columnas xG en football-data.co.uk (cuando existen)
COL_XG_HOME = "HxG"
COL_XG_AWAY = "AxG"
COL_GOLES_HOME = "FTHG"   # Full Time Home Goals
COL_GOLES_AWAY = "FTAG"   # Full Time Away Goals
COL_EQUIPO_HOME = "HomeTeam"
COL_EQUIPO_AWAY = "AwayTeam"

PARTIDOS_RECIENTES = 5   # cuántos partidos recientes usar para el promedio


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

@dataclass
class DatosXG:
    """Datos de xG de un equipo en los últimos N partidos."""
    equipo: str
    liga: str
    partidos: int                   # cuántos partidos se encontraron (puede ser < 5)

    xg_promedio: float              # xG generado promedio
    xga_promedio: float             # xG recibido promedio
    goles_promedio: float           # goles reales promedio
    goles_recibidos_promedio: float

    # Diferencia acumulada xG vs goles reales (últimos N partidos)
    # Positivo = equipo está marcando MENOS de lo que sus xG indican (candidato a regresar)
    # Negativo = equipo está marcando MÁS de lo que sus xG indican (puede corregir a la baja)
    diferencia_xg_goles: float

    flag_regresion: bool            # True si abs(diferencia) > 1.5 en últimos 5 partidos
    direccion_regresion: str        # "SUBE" (esperamos más goles) | "BAJA" | "NEUTRAL"

    def resumen(self) -> str:
        """Texto compacto para incluir en el prompt de Claude."""
        lines = [
            f"xG promedio (ataque): {self.xg_promedio:.2f} | xGA promedio (defensa): {self.xga_promedio:.2f}",
            f"Goles reales promedio: {self.goles_promedio:.2f}",
            f"Diferencia xG−goles: {self.diferencia_xg_goles:+.2f}",
        ]
        if self.flag_regresion:
            lines.append(f"⚠️  REGRESIÓN A LA MEDIA: equipo debería {'MARCAR MÁS' if self.direccion_regresion == 'SUBE' else 'MARCAR MENOS'} próximamente")
        return " | ".join(lines)


# ─────────────────────────────────────────────────────────────
# Normalización de nombres de equipos
# ─────────────────────────────────────────────────────────────

def _normalizar(nombre: str) -> str:
    """Lowercase, sin tildes, sin caracteres especiales."""
    nombre = nombre.lower().strip()
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    nombre = re.sub(r"[^a-z0-9 ]", "", nombre)
    nombre = re.sub(r"\s+", " ", nombre).strip()
    return nombre


def _similitud(a: str, b: str) -> float:
    """
    Similitud simple: fracción de palabras de 'a' que aparecen en 'b'.
    Suficiente para mapear "Man City" ↔ "Manchester City".
    """
    palabras_a = set(_normalizar(a).split())
    palabras_b = set(_normalizar(b).split())
    if not palabras_a:
        return 0.0
    return len(palabras_a & palabras_b) / len(palabras_a)


def _buscar_equipo_en_df(nombre_buscado: str, equipos_en_df: list[str]) -> Optional[str]:
    """
    Devuelve el nombre exacto del equipo en el DataFrame que mejor matchea
    con el nombre buscado. Umbral mínimo de similitud: 0.5.
    """
    normalizado = _normalizar(nombre_buscado)
    mejor_match = None
    mejor_score = 0.0

    for eq in equipos_en_df:
        score = _similitud(normalizado, _normalizar(eq))
        if score > mejor_score:
            mejor_score = score
            mejor_match = eq

    if mejor_score >= 0.5:
        return mejor_match
    return None


# ─────────────────────────────────────────────────────────────
# Fetch y caché del CSV
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _fetch_csv(url: str) -> Optional[pd.DataFrame]:
    """
    Descarga el CSV y lo devuelve como DataFrame.
    Cacheado en memoria para no re-fetchear en cada ciclo.
    lru_cache se resetea cuando el proceso reinicia (Render reinicia el worker).
    """
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="latin-1")
        logger.debug(f"CSV cargado: {url} — {len(df)} filas, columnas: {list(df.columns)[:10]}")
        return df
    except Exception as e:
        logger.warning(f"No se pudo fetchear CSV {url}: {e}")
        return None


def _tiene_xg(df: pd.DataFrame) -> bool:
    """Verifica si el CSV tiene columnas xG."""
    return COL_XG_HOME in df.columns and COL_XG_AWAY in df.columns


# ─────────────────────────────────────────────────────────────
# Cálculo de xG por equipo
# ─────────────────────────────────────────────────────────────

def _calcular_xg_equipo(df: pd.DataFrame, nombre_equipo: str, n: int = PARTIDOS_RECIENTES) -> Optional[DatosXG]:
    """
    Busca los últimos N partidos del equipo (como local o visitante)
    y calcula sus métricas de xG.
    """
    # Obtener lista de equipos únicos para el matching
    equipos = list(set(
        list(df[COL_EQUIPO_HOME].dropna().unique()) +
        list(df[COL_EQUIPO_AWAY].dropna().unique())
    ))

    nombre_exacto = _buscar_equipo_en_df(nombre_equipo, equipos)
    if not nombre_exacto:
        logger.debug(f"Equipo no encontrado en CSV: {nombre_equipo}")
        return None

    # Partidos como local
    df_home = df[df[COL_EQUIPO_HOME] == nombre_exacto].copy()
    df_home["es_local"] = True
    df_home["xg_favor"] = df_home[COL_XG_HOME] if _tiene_xg(df) else None
    df_home["xg_contra"] = df_home[COL_XG_AWAY] if _tiene_xg(df) else None
    df_home["goles_favor"] = df_home[COL_GOLES_HOME]
    df_home["goles_contra"] = df_home[COL_GOLES_AWAY]

    # Partidos como visitante
    df_away = df[df[COL_EQUIPO_AWAY] == nombre_exacto].copy()
    df_away["es_local"] = False
    df_away["xg_favor"] = df_away[COL_XG_AWAY] if _tiene_xg(df) else None
    df_away["xg_contra"] = df_away[COL_XG_HOME] if _tiene_xg(df) else None
    df_away["goles_favor"] = df_away[COL_GOLES_AWAY]
    df_away["goles_contra"] = df_away[COL_GOLES_HOME]

    # Combinar y tomar los últimos N
    cols = ["xg_favor", "xg_contra", "goles_favor", "goles_contra"]
    df_equipo = pd.concat([
        df_home[cols],
        df_away[cols],
    ]).tail(n)

    if len(df_equipo) == 0:
        return None

    tiene_xg_data = _tiene_xg(df) and df_equipo["xg_favor"].notna().any()

    if tiene_xg_data:
        xg_prom = float(df_equipo["xg_favor"].mean())
        xga_prom = float(df_equipo["xg_contra"].mean())
        goles_prom = float(df_equipo["goles_favor"].mean())
        goles_rec_prom = float(df_equipo["goles_contra"].mean())

        # Diferencia xG − goles en los últimos N partidos (acumulado)
        dif = float((df_equipo["xg_favor"] - df_equipo["goles_favor"]).sum())
    else:
        # Sin xG disponible: usar solo estadísticas de goles
        xg_prom = float(df_equipo["goles_favor"].mean())   # fallback
        xga_prom = float(df_equipo["goles_contra"].mean())
        goles_prom = xg_prom
        goles_rec_prom = xga_prom
        dif = 0.0

    # Regresión a la media
    flag_regresion = abs(dif) > 1.5
    if dif > 1.5:
        direccion = "SUBE"    # marcó menos de lo esperado → va a subir
    elif dif < -1.5:
        direccion = "BAJA"    # marcó más de lo esperado → va a bajar
    else:
        direccion = "NEUTRAL"

    return DatosXG(
        equipo=nombre_exacto,
        liga="",
        partidos=len(df_equipo),
        xg_promedio=xg_prom,
        xga_promedio=xga_prom,
        goles_promedio=goles_prom,
        goles_recibidos_promedio=goles_rec_prom,
        diferencia_xg_goles=dif,
        flag_regresion=flag_regresion,
        direccion_regresion=direccion,
    )


# ─────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────

def _detectar_liga(titulo_partido: str, liga_azuro: str) -> Optional[str]:
    """
    Intenta detectar qué clave de LIGA_URLS corresponde al partido.
    Usa la liga de Azuro y el título del partido.
    """
    texto = f"{titulo_partido} {liga_azuro}".lower()

    mapping = {
        "premier": "Premier League",
        "england. premier": "Premier League",
        "championship": "Championship",
        "la liga": "La Liga",
        "spain": "La Liga",
        "bundesliga 2": "Bundesliga 2",
        "bundesliga": "Bundesliga",
        "serie a": "Serie A",
        "italy": "Serie A",
        "ligue 1": "Ligue 1",
        "france": "Ligue 1",
        "eredivisie": "Eredivisie",
        "netherlands": "Eredivisie",
        "liga portugal": "Liga Portugal",
        "portugal": "Liga Portugal",
    }

    for keyword, liga_key in mapping.items():
        if keyword in texto:
            return liga_key

    return None


def get_xg_partido(
    titulo_partido: str,
    liga_azuro: str,
    equipo_home: str,
    equipo_away: str,
) -> tuple[Optional[DatosXG], Optional[DatosXG]]:
    """
    Devuelve (xg_home, xg_away) para un partido.
    Si no se puede obtener la data, devuelve (None, None).
    """
    liga_key = _detectar_liga(titulo_partido, liga_azuro)
    if not liga_key:
        logger.debug(f"Liga no soportada para xG: {liga_azuro}")
        return None, None

    url = LIGA_URLS[liga_key]
    df = _fetch_csv(url)
    if df is None:
        return None, None

    xg_home = _calcular_xg_equipo(df, equipo_home)
    xg_away = _calcular_xg_equipo(df, equipo_away)

    if xg_home:
        xg_home.liga = liga_key
    if xg_away:
        xg_away.liga = liga_key

    return xg_home, xg_away


def _extraer_equipos(titulo: str) -> tuple[str, str]:
    """
    Extrae nombres de equipos desde el título del partido.
    Formato esperado: "Team A vs Team B" o "Team A - Team B"
    """
    for sep in [" vs ", " - ", " v "]:
        if sep in titulo:
            partes = titulo.split(sep, 1)
            return partes[0].strip(), partes[1].strip()
    # Fallback
    return titulo, ""


def enriquecer_mercado_con_xg(titulo: str, liga: str) -> dict:
    """
    Dado el título de un partido y su liga (como vienen de Azuro),
    devuelve un dict con los datos xG para incluir en el prompt de Claude.

    Retorna:
    {
        "disponible": bool,
        "home": DatosXG | None,
        "away": DatosXG | None,
        "flag_regresion_home": bool,
        "flag_regresion_away": bool,
        "score_bonus": float,   # multiplicador de score si hay regresión (1.0 o 1.3)
        "resumen": str          # texto para el prompt
    }
    """
    equipo_home, equipo_away = _extraer_equipos(titulo)
    xg_home, xg_away = get_xg_partido(titulo, liga, equipo_home, equipo_away)

    disponible = xg_home is not None or xg_away is not None
    flag_h = xg_home.flag_regresion if xg_home else False
    flag_a = xg_away.flag_regresion if xg_away else False

    # Score bonus si hay regresión en alguno de los dos equipos
    score_bonus = 1.3 if (flag_h or flag_a) else 1.0

    if not disponible:
        return {
            "disponible": False,
            "home": None,
            "away": None,
            "flag_regresion_home": False,
            "flag_regresion_away": False,
            "score_bonus": 1.0,
            "resumen": "Datos xG no disponibles para esta liga.",
        }

    lineas = ["📊 DATOS xG (últimos 5 partidos):"]
    if xg_home:
        lineas.append(f"  LOCAL ({xg_home.equipo}): {xg_home.resumen()}")
    if xg_away:
        lineas.append(f"  VISITANTE ({xg_away.equipo}): {xg_away.resumen()}")

    return {
        "disponible": True,
        "home": xg_home,
        "away": xg_away,
        "flag_regresion_home": flag_h,
        "flag_regresion_away": flag_a,
        "score_bonus": score_bonus,
        "resumen": "\n".join(lineas),
    }


# ─────────────────────────────────────────────────────────────
# Testing — python -m bot.xg_data
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    casos_test = [
        ("Arsenal vs Chelsea", "Premier League", "Arsenal", "Chelsea"),
        ("Ajax vs PSV", "Eredivisie", "Ajax", "PSV"),
        ("Barcelona vs Real Madrid", "La Liga", "Barcelona", "Real Madrid"),
        ("Fulham vs Brentford", "Championship", "Fulham", "Brentford"),
    ]

    for titulo, liga, home, away in casos_test:
        print(f"\n{'='*60}")
        print(f"PARTIDO: {titulo}")
        xg_h, xg_a = get_xg_partido(titulo, liga, home, away)

        if xg_h:
            print(f"  LOCAL: {xg_h.resumen()}")
            print(f"  Regresión: {xg_h.flag_regresion} ({xg_h.direccion_regresion})")
        else:
            print(f"  LOCAL: sin datos xG")

        if xg_a:
            print(f"  VISITANTE: {xg_a.resumen()}")
            print(f"  Regresión: {xg_a.flag_regresion} ({xg_a.direccion_regresion})")
        else:
            print(f"  VISITANTE: sin datos xG")
