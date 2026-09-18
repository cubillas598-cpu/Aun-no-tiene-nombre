"""Modelos de inercia y teorema de Steiner."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inercia_varilla(m, L):
    """Varilla delgada con su eje largo sobre x."""
    return np.diag([0.0, m * L ** 2 / 12, m * L ** 2 / 12])


def inercia_cilindro(m, r, L):
    """Cilindro con su eje sobre x."""
    return np.diag([m * r ** 2 / 2, m * (3 * r ** 2 + L ** 2) / 12,
                    m * (3 * r ** 2 + L ** 2) / 12])


def inercia_prisma(m, a, b, c):
    return np.diag([m * (b ** 2 + c ** 2) / 12, m * (a ** 2 + c ** 2) / 12,
                    m * (a ** 2 + b ** 2) / 12])


def inercia_esfera(m, r):
    return np.diag([2 * m * r ** 2 / 5] * 3)


FORMAS = {
    "Varilla delgada (eje en x)": (inercia_varilla, ["largo L [m]"]),
    "Cilindro (eje en x)": (inercia_cilindro, ["radio r [m]", "largo L [m]"]),
    "Prisma rectangular": (inercia_prisma, ["lado a [m]", "lado b [m]", "lado c [m]"]),
    "Esfera maciza": (inercia_esfera, ["radio r [m]"]),
}


def steiner(I: np.ndarray, m: float, d: Sequence[float]) -> np.ndarray:
    """Teorema de los ejes paralelos."""
    d = np.asarray(d, float)
    return I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
