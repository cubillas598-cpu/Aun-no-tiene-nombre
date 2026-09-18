"""Estado de sesion y conversiones de unidades de la interfaz."""

from __future__ import annotations

import numpy as np
import streamlit as st

from domain.catalog import catalogo
from domain.robot import Eslabon
from domain.rotations import deg, rad


def ss():
    return st.session_state


def _cargar(nombre: str) -> None:
    robot, q0 = catalogo()[nombre]()
    ss().robot = robot
    ss().q = np.asarray(q0, float)
    ss().nombre = nombre
    ss()._elegido = nombre
    ss().version = ss().get("version", 0) + 1
    ss().nube = None


def iniciar_estado() -> None:
    if "robot" not in ss():
        ss().unidad_ang = "grados"
        ss().ver_explicaciones = True
        ss().grados = True
        ss().explicar = True
        _cargar("Brazo antropomorfico (3 ejes)")
    ss().grados = ss().get("unidad_ang", "grados") == "grados"
    ss().explicar = bool(ss().get("ver_explicaciones", True))


def uni() -> str:
    return "°" if ss().grados else "rad"


def a_pantalla(valor: float) -> float:
    return deg(valor) if ss().grados else valor


def a_interno(valor: float) -> float:
    return rad(valor) if ss().grados else valor


def unidad_junta(eslabon: Eslabon) -> str:
    return uni() if eslabon.es_rotacion else "m"


def unidad_par(eslabon: Eslabon) -> str:
    return "N·m" if eslabon.es_rotacion else "N"


def n_fmt(x: float, dec: int = 3) -> str:
    x = float(x)
    if abs(x) < 5e-7:
        x = 0.0
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{dec}f}"


def latex_matriz(M: np.ndarray, nombre: str = "", dec: int = 3) -> str:
    cuerpo = r" \\ ".join(" & ".join(n_fmt(v, dec) for v in fila) for fila in np.atleast_2d(M))
    izq = f"{nombre} = " if nombre else ""
    return izq + r"\begin{bmatrix}" + cuerpo + r"\end{bmatrix}"
