"""Pestana de cinematica directa."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from domain.robot import Robot
from domain.rotations import (cuaternion_desde_R, eje_angulo_desde_R, rpy_desde_R,
                              zyz_desde_R)
from ui.components import dice, encabezado, pista, tabla
from ui.state import a_pantalla, latex_matriz, n_fmt, ss, uni


def tab_directa() -> None:
    rob: Robot = ss().robot
    q = ss().q
    Ts = rob.cadena(q)
    T = Ts[-1]
    p = T[:3, 3]
    encabezado("¿Dónde está la mano?",
               "Le das los ángulos de cada motor y te dice en qué punto del espacio queda el extremo.")
    dice(f"Con la postura de ahora, la mano está en <b>x = {p[0]:.3f} m</b>, "
         f"<b>y = {p[1]:.3f} m</b>, <b>z = {p[2]:.3f} m</b>. "
         f"Todo eso está guardado en una sola matriz de 4×4.")
    pista("Se multiplica una matriz por eslabón: <b>T = A₁·A₂·…·Aₙ</b>. En el resultado, la última columna "
          "es la <b>posición</b> de la mano y el bloque de 3×3 de la izquierda es su <b>orientación</b>, "
          "o sea hacia dónde apunta.")

    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("**La matriz que lo resume todo**")
        st.latex(latex_matriz(T, "T", 3))
    with der:
        st.markdown("**La misma orientación, escrita de todas las formas**")
        roll, pitch, yaw = rpy_desde_R(T[:3, :3])
        phi, tht, psi = zyz_desde_R(T[:3, :3])
        eje, ang = eje_angulo_desde_R(T[:3, :3])
        cu = cuaternion_desde_R(T[:3, :3])
        tabla(pd.DataFrame({
            "Forma": ["Posición [m]", f"RPY fijos ZYX [{uni()}]", f"Euler ZYZ [{uni()}]",
                      f"Eje y ángulo [{uni()}]", "Cuaternión"],
            "Valores": [
                f"x {n_fmt(p[0])}   y {n_fmt(p[1])}   z {n_fmt(p[2])}",
                f"{n_fmt(a_pantalla(roll), 2)}   {n_fmt(a_pantalla(pitch), 2)}   {n_fmt(a_pantalla(yaw), 2)}",
                f"{n_fmt(a_pantalla(phi), 2)}   {n_fmt(a_pantalla(tht), 2)}   {n_fmt(a_pantalla(psi), 2)}",
                f"eje ({', '.join(n_fmt(v, 2) for v in eje)})   ángulo {n_fmt(a_pantalla(ang), 2)}",
                "   ".join(n_fmt(v) for v in cu)]}))

    with st.expander("Ver el procedimiento eslabón por eslabón (para copiar a la tarea)"):
        for i, e in enumerate(rob.eslabones):
            th = e.theta + q[i] if e.es_rotacion else e.theta
            dd = e.d if e.es_rotacion else e.d + q[i]
            extra = " (el desplazamiento fijo más q)" if (e.es_rotacion and e.theta) else (" = q" if e.es_rotacion else "")
            st.markdown(f"**Eslabón {i + 1}** · θ = {n_fmt(a_pantalla(th), 2)}{uni()}{extra} · "
                        f"d = {n_fmt(dd)} m · a = {n_fmt(e.a)} m · α = {n_fmt(a_pantalla(e.alpha), 2)}{uni()}")
            c1, c2 = st.columns(2)
            with c1:
                st.latex(latex_matriz(e.matriz(float(q[i])), f"A_{{{i + 1}}}", 3))
            with c2:
                st.latex(latex_matriz(Ts[i + 1], f"T^0_{{{i + 1}}}", 3))
        st.code("\n".join(" ".join(f"{v:10.5f}" for v in fila) for fila in T), language=None)
