"""
agent/sessions.py
Lógica de sesiones de trading basada en horarios de Argentina (America/Argentina/Buenos_Aires).
Detecta en qué sesión estamos y qué prioridad tiene operar en ese momento.

Modo HÍBRIDO:
- EVITAR  → bloquea operación (baja liquidez)
- BAJA    → permite operar pero el agente debe ser muy selectivo
- MEDIA   → operación normal
- ALTA    → operación con prioridad (más capital permitido por apuesta)
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from enum import Enum


TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")


class Prioridad(str, Enum):
    EVITAR = "EVITAR"
    BAJA = "BAJA"
    MEDIA = "MEDIA"
    ALTA = "ALTA"


@dataclass
class Sesion:
    nombre: str
    prioridad: Prioridad
    descripcion: str
    puede_operar: bool

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "prioridad": self.prioridad.value,
            "descripcion": self.descripcion,
            "puede_operar": self.puede_operar,
        }


# Definición de ventanas horarias (hora Argentina UTC-3)
# Formato: (hora_inicio, hora_fin, nombre, prioridad, descripción)
# Las ventanas se evalúan en orden — la primera que matchea gana.
# Por eso las más específicas (NY Kill dentro de Overlap) van PRIMERO.
VENTANAS = [
    # Ventanas específicas primero (más prioritarias)
    (time(10, 30), time(12, 0), "NY_KILL_ZONE", Prioridad.ALTA,
     "NY Kill Zone — alta volatilidad por apertura USA"),

    (time(9, 0), time(13, 0), "OVERLAP_LONDON_NY", Prioridad.ALTA,
     "Overlap London-NY — máximo volumen y liquidez del día"),

    (time(5, 0), time(8, 0), "LONDON_KILL_ZONE", Prioridad.ALTA,
     "London Kill Zone — caza de stops del rango asiático"),

    # Ventanas amplias después
    (time(8, 0), time(9, 0), "PRE_OVERLAP", Prioridad.MEDIA,
     "Transición London a Overlap"),

    (time(13, 0), time(14, 0), "POST_OVERLAP", Prioridad.MEDIA,
     "Cierre del overlap, aún hay liquidez"),

    (time(14, 0), time(21, 0), "BAJA_LIQUIDEZ", Prioridad.EVITAR,
     "Tarde argentina — baja liquidez, NO operar"),

    # Sesión asiática (cruza medianoche): 21:00 → 05:00
    # Se maneja aparte por el wrap.
]


def _esta_en_sesion_asiatica(hora: time) -> bool:
    """La sesión asiática cruza medianoche: 21:00 a 05:00 hs ARG."""
    return hora >= time(21, 0) or hora < time(5, 0)


def get_sesion_actual(now: datetime | None = None) -> Sesion:
    """
    Devuelve la sesión activa en este momento (hora Argentina).

    Args:
        now: datetime opcional (útil para testing). Si es None usa ahora.
             Si viene sin tzinfo, se asume que ya está en hora ARG.

    Returns:
        Sesion con nombre, prioridad, descripción y flag puede_operar.
    """
    if now is None:
        now = datetime.now(TZ_ARG)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=TZ_ARG)
    else:
        now = now.astimezone(TZ_ARG)

    hora_actual = now.time()

    # 1. Chequear sesión asiática (caso especial: cruza medianoche)
    if _esta_en_sesion_asiatica(hora_actual):
        return Sesion(
            nombre="ASIATICA",
            prioridad=Prioridad.EVITAR,
            descripcion="Sesión asiática — solo monitorear, NO operar (define rango del día)",
            puede_operar=False,
        )

    # 2. Chequear ventanas en orden (la primera que matchea gana)
    for inicio, fin, nombre, prioridad, descripcion in VENTANAS:
        if inicio <= hora_actual < fin:
            return Sesion(
                nombre=nombre,
                prioridad=prioridad,
                descripcion=descripcion,
                puede_operar=(prioridad != Prioridad.EVITAR),
            )

    # 3. Fallback (no debería pasar, pero por seguridad)
    return Sesion(
        nombre="DESCONOCIDA",
        prioridad=Prioridad.EVITAR,
        descripcion="Horario sin clasificar — por seguridad, no operar",
        puede_operar=False,
    )


def puede_operar_ahora(now: datetime | None = None) -> bool:
    """Helper rápido: ¿el bot puede operar en este momento?"""
    return get_sesion_actual(now).puede_operar


def get_multiplicador_capital(prioridad: Prioridad) -> float:
    """
    Devuelve el multiplicador de tamaño de apuesta según prioridad de sesión.
    Se aplica sobre el % base definido en risk.py (3-5% para Nivel 1).

    ALTA  → 1.0  (usa el % base completo)
    MEDIA → 0.7  (reduce 30% el tamaño)
    BAJA  → 0.5  (mitad del tamaño base)
    EVITAR → 0.0 (no opera)
    """
    return {
        Prioridad.ALTA: 1.0,
        Prioridad.MEDIA: 0.7,
        Prioridad.BAJA: 0.5,
        Prioridad.EVITAR: 0.0,
    }[prioridad]


# ─────────────────────────────────────────────────────────────
# Testing rápido — correr con: python -m agent.sessions
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Sesiones de trading (hora Argentina)")
    print("=" * 60)

    # Test sesión actual
    sesion = get_sesion_actual()
    ahora = datetime.now(TZ_ARG)
    print(f"\n🕐 Ahora: {ahora.strftime('%H:%M:%S')} hs ARG")
    print(f"📍 Sesión: {sesion.nombre}")
    print(f"⚡ Prioridad: {sesion.prioridad.value}")
    print(f"📝 {sesion.descripcion}")
    print(f"✅ Puede operar: {sesion.puede_operar}")
    print(f"💰 Multiplicador capital: {get_multiplicador_capital(sesion.prioridad)}x")

    # Test horarios clave
    print("\n" + "=" * 60)
    print("TEST: Horarios clave del día")
    print("=" * 60)
    horarios_test = [
        (3, 0, "Madrugada (asiática)"),
        (6, 0, "London Kill Zone"),
        (8, 30, "Pre-overlap"),
        (10, 0, "Overlap London-NY"),
        (11, 0, "NY Kill Zone (dentro de overlap)"),
        (13, 30, "Post-overlap"),
        (16, 0, "Tarde argentina"),
        (22, 0, "Noche (asiática)"),
    ]
    for h, m, desc in horarios_test:
        test_dt = datetime.now(TZ_ARG).replace(hour=h, minute=m, second=0, microsecond=0)
        s = get_sesion_actual(test_dt)
        flag = "✅" if s.puede_operar else "❌"
        print(f"{flag} {h:02d}:{m:02d} → {s.nombre:20s} [{s.prioridad.value:6s}] {desc}")