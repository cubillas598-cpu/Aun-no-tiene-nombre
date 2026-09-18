"""Perfiles de movimiento articular."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def cubica(q0, qf, v0, vf, tf) -> Callable[[float], np.ndarray]:
    """Perfil cubico con velocidad continua."""
    a0, a1 = q0, v0
    a2 = (3 * (qf - q0) - (2 * v0 + vf) * tf) / tf ** 2
    a3 = (2 * (q0 - qf) + (v0 + vf) * tf) / tf ** 3

    def f(t):
        return np.array([a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3,
                         a1 + 2 * a2 * t + 3 * a3 * t ** 2,
                         2 * a2 + 6 * a3 * t])
    return f


def quintica(q0, qf, v0, vf, ac0, acf, tf) -> Callable[[float], np.ndarray]:
    """Perfil quintico con aceleracion inicial y final controladas."""
    d, T = qf - q0, tf
    a0, a1, a2 = q0, v0, ac0 / 2
    a3 = (20 * d - (8 * vf + 12 * v0) * T - (3 * ac0 - acf) * T ** 2) / (2 * T ** 3)
    a4 = (-30 * d + (14 * vf + 16 * v0) * T + (3 * ac0 - 2 * acf) * T ** 2) / (2 * T ** 4)
    a5 = (12 * d - 6 * (vf + v0) * T + (acf - ac0) * T ** 2) / (2 * T ** 5)

    def f(t):
        return np.array([a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5,
                         a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4,
                         2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3])
    return f


def trapezoidal(q0, qf, tf, aceleracion) -> Callable[[float], np.ndarray]:
    """Perfil trapezoidal de velocidad."""
    d = qf - q0
    D = abs(d)
    signo = math.copysign(1.0, d) if d else 1.0
    if D < 1e-12:
        f = lambda t: np.array([q0, 0.0, 0.0])
        f.tc, f.v_crucero, f.aceleracion, f.insuficiente = 0.0, 0.0, 0.0, False
        return f
    minima = 4 * D / tf ** 2
    insuficiente = aceleracion < minima
    a = max(aceleracion, minima * 1.0000001)
    tc = tf / 2 - math.sqrt(max(a * a * tf * tf - 4 * a * D, 0.0)) / (2 * a)
    vc = a * tc

    def f(t):
        t = min(max(t, 0.0), tf)
        if t <= tc:
            return np.array([q0 + signo * 0.5 * a * t * t, signo * a * t, signo * a])
        if t <= tf - tc:
            return np.array([q0 + signo * vc * (t - tc / 2), signo * vc, 0.0])
        u = tf - t
        return np.array([qf - signo * 0.5 * a * u * u, signo * a * u, -signo * a])

    f.tc, f.v_crucero, f.aceleracion, f.insuficiente = tc, signo * vc, signo * a, insuficiente
    return f
