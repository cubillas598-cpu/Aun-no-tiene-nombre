"""Pestana de conversion y composicion de rotaciones."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from domain.rotations import (R_desde_cuaternion, R_desde_eje_angulo, R_desde_rpy,
                              R_desde_zyz, cuaternion_desde_R, eje_angulo_desde_R,
                              homogenea, inversa_homogenea, rpy_desde_R, rot_x, rot_y,
                              rot_z, zyz_desde_R)
from ui.components import dice, pista, tabla
from ui.state import a_interno, a_pantalla, latex_matriz, n_fmt, uni


def tab_rotaciones(vector_entrada) -> None:
    pista("Los <b>ángulos</b> son fáciles de leer pero tienen posturas donde se pierde información "
          "(el famoso bloqueo de cardán). El <b>cuaternión</b> no tiene ese problema y por eso es el que "
          "se usa para interpolar giros.")
    st.markdown("### Convertir entre todas las formas de escribir una orientación")
    st.markdown("La misma orientación se puede escribir de cinco maneras. Aquí pasas de una a otra.")

    tipo = st.selectbox("¿Cómo vas a dar la orientación?",
                        ["Ángulos fijos RPY ZYX (roll, pitch, yaw)", "Ángulos de Euler móviles ZYZ",
                         "Eje y ángulo", "Cuaternión", "Matriz de rotación"])
    if tipo.startswith("Ángulos fijos"):
        v = vector_entrada([f"roll [{uni()}]", f"pitch [{uni()}]", f"yaw [{uni()}]"], [0, 0, 0], "rr", 5.0)
        R = R_desde_rpy(*[a_interno(x) for x in v])
    elif tipo.startswith("Ángulos de Euler"):
        v = vector_entrada([f"φ [{uni()}]", f"θ [{uni()}]", f"ψ [{uni()}]"], [0, 0, 0], "rz", 5.0)
        R = R_desde_zyz(*[a_interno(x) for x in v])
    elif tipo.startswith("Eje"):
        v = vector_entrada(["eje x", "eje y", "eje z", f"ángulo [{uni()}]"], [0, 0, 1, 0], "ra", 0.1)
        R = R_desde_eje_angulo(v[:3], a_interno(v[3]))
    elif tipo.startswith("Cuaternión"):
        v = vector_entrada(["w", "x", "y", "z"], [1, 0, 0, 0], "rq", 0.05)
        R = R_desde_cuaternion(v)
    else:
        st.caption("Escribe los nueve números de la matriz:")
        filas = [vector_entrada([f"fila {i + 1} · col {j + 1}" for j in range(3)],
                                [1.0 if i == j else 0.0 for j in range(3)], f"rm{i}", 0.1)
                 for i in range(3)]
        R = np.array(filas, float)

    desvio = float(np.max(np.abs(R @ R.T - np.eye(3))))
    if desvio > 1e-6:
        st.warning(f"Esa matriz no es una rotación válida: se aparta {desvio:.4f} de cumplir R·Rᵀ = I "
                   f"(sus columnas deberían ser perpendiculares y de largo 1). "
                   f"Las conversiones de abajo saldrán aproximadas.", icon="⚠️")
    p = vector_entrada(["px [m]", "py [m]", "pz [m]"], [0, 0, 0], "tp", 0.05)
    T = homogenea(R, p)
    eje, ang = eje_angulo_desde_R(R)
    roll, pitch, yaw = rpy_desde_R(R)
    phi, tht, psi = zyz_desde_R(R)
    cu = cuaternion_desde_R(R)
    if desvio <= 1e-6:
        dice(f"Esa orientación equivale a girar <b>{n_fmt(a_pantalla(ang), 2)} {uni()}</b> alrededor "
             f"de un solo eje: ({', '.join(n_fmt(v, 3) for v in eje)}).")

    c1, c2 = st.columns(2)
    with c1:
        st.latex(latex_matriz(R, "R", 4))
        st.latex(latex_matriz(T, "T", 4))
    with c2:
        st.latex(latex_matriz(inversa_homogenea(T), "T^{-1}", 4))
        tabla(pd.DataFrame({
            "Forma": [f"RPY ZYX [{uni()}]", f"Euler ZYZ [{uni()}]", f"Eje y ángulo [{uni()}]",
                      "Cuaternión", "det(R)"],
            "Valores": [
                "   ".join(n_fmt(a_pantalla(x), 2) for x in (roll, pitch, yaw)),
                "   ".join(n_fmt(a_pantalla(x), 2) for x in (phi, tht, psi)),
                f"({', '.join(n_fmt(v, 3) for v in eje)})   {n_fmt(a_pantalla(ang), 2)}",
                "   ".join(n_fmt(v, 4) for v in cu),
                n_fmt(float(np.linalg.det(R)), 5)]}))

    cardan = ((tipo.startswith("Ángulos fijos") and abs(abs(pitch) - math.pi / 2) < 1e-3)
              or (tipo.startswith("Ángulos de Euler") and abs(math.sin(tht)) < 1e-3))
    if cardan:
        st.warning("Estás justo en una postura degenerada de estos ángulos: dos de los tres giros hacen "
                   "lo mismo y el reparto entre ellos ya no es único. Es el bloqueo de cardán; "
                   "con eje-ángulo o con cuaternión no pasa.", icon="⚠️")

    st.divider()
    st.markdown("#### Encadenar varios movimientos")
    pista("Si cada giro se hace sobre los <b>ejes que se mueven</b> junto con el objeto, las matrices se "
          "multiplican en el mismo orden en que ocurren. Si todos se miden respecto al <b>marco fijo</b> "
          "del piso, se multiplican en orden inverso. Ése es el error más común del curso.")
    marco = st.radio("¿Respecto a qué se mide cada paso?", ["Ejes móviles", "Marco fijo"], horizontal=True)
    pasos_df = st.data_editor(
        pd.DataFrame({"Movimiento": ["Girar sobre z", "Mover en x"], "Cuánto": [90.0, 0.5]}),
        num_rows="dynamic", hide_index=True, key="comp_pasos",
        column_config={"Movimiento": st.column_config.SelectboxColumn(
            options=["Girar sobre x", "Girar sobre y", "Girar sobre z",
                     "Mover en x", "Mover en y", "Mover en z"], required=True),
            "Cuánto": st.column_config.NumberColumn(help=f"grados si es un giro ({uni()}), metros si es mover",
                                                    format="%.3f")})
    Tc = np.eye(4)
    nombres = []
    for k, fila in pasos_df.iterrows():
        mov, cuanto = str(fila["Movimiento"]), float(fila["Cuánto"] or 0.0)
        if mov.startswith("Girar"):
            ang_k = a_interno(cuanto)
            R_k = {"x": rot_x, "y": rot_y, "z": rot_z}[mov[-1]](ang_k)
            A_k = homogenea(R_k, [0, 0, 0])
        else:
            p_k = np.zeros(3)
            p_k["xyz".index(mov[-1])] = cuanto
            A_k = homogenea(np.eye(3), p_k)
        Tc = Tc @ A_k if marco == "Ejes móviles" else A_k @ Tc
        nombres.append(f"A{k + 1}")
    orden = " · ".join(nombres if marco == "Ejes móviles" else list(reversed(nombres)))
    dice(f"Con <b>{marco.lower()}</b>, el producto queda <b>T = {orden}</b>.")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(latex_matriz(Tc, "T", 4))
    with c2:
        rc = rpy_desde_R(Tc[:3, :3])
        tabla(pd.DataFrame({"Resultado": ["posición [m]", f"RPY ZYX [{uni()}]"],
                            "Valores": ["   ".join(n_fmt(v) for v in Tc[:3, 3]),
                                        "   ".join(n_fmt(a_pantalla(v), 2) for v in rc)]}))
