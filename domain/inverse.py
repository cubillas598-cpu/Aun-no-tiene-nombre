"""Soluciones cerradas de cinematica inversa para geometrías concretas."""

from __future__ import annotations

import math

import numpy as np


def inversa_2R(a1: float, a2: float, x: float, y: float) -> list[dict]:
    """Brazo plano de dos eslabones: devuelve codo arriba y abajo."""
    r2 = x * x + y * y
    c2 = (r2 - a1 * a1 - a2 * a2) / (2 * a1 * a2) if a1 * a2 else 2.0
    if abs(c2) > 1 + 1e-9:
        return []
    c2 = max(-1.0, min(1.0, c2))
    s2 = math.sqrt(max(1 - c2 * c2, 0.0))
    sols = []
    for signo, nombre in ((1, "arriba"), (-1, "abajo")):
        th2 = math.atan2(signo * s2, c2)
        th1 = math.atan2(y, x) - math.atan2(a2 * math.sin(th2), a1 + a2 * math.cos(th2))
        sols.append({"q": np.array([th1, th2]), "codo": nombre})
    return sols


def inversa_3R_plano(a1, a2, a3, x, y, phi) -> list[dict]:
    """Brazo plano de tres eslabones con orientación final phi."""
    xm = x - a3 * math.cos(phi)
    ym = y - a3 * math.sin(phi)
    out = []
    for s in inversa_2R(a1, a2, xm, ym):
        th1, th2 = s["q"]
        out.append({"q": np.array([th1, th2, phi - th1 - th2]), "codo": s["codo"]})
    return out


def inversa_antropomorfico(a1, a2, a3, d1, x, y, z) -> list[dict]:
    """Brazo antropomórfico de tres ejes: hasta cuatro soluciones."""
    out = []
    for atras in (False, True):
        th1 = math.atan2(y, x) + (math.pi if atras else 0.0)
        r = (-math.hypot(x, y) if atras else math.hypot(x, y)) - a1
        s = z - d1
        for sol in inversa_2R(a2, a3, r, s):
            out.append({"q": np.array([th1, sol["q"][0], sol["q"][1]]),
                        "codo": sol["codo"], "hombro": "atras" if atras else "adelante"})
    return out
