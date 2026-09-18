"""Pestana de jacobiano, velocidades y singularidades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from domain.robot import Robot
from ui.components import dice, encabezado, pista, tabla
from ui.state import latex_matriz, n_fmt, ss, unidad_par


def tab_jacobiano(vector_entrada, argumentos_tabla: dict | None = None) -> None:
    rob: Robot = ss().robot
    q = ss().q
    n = rob.n
    J = rob.jacobiano(q)
    sv = rob.valores_singulares(q)
    rango = rob.rango(q)
    w = rob.manipulabilidad(q)
    completo = min(6, n)
    singular = rango < completo

    encabezado("¿Cómo se mueve la mano y dónde se atora el robot?",
               "El jacobiano es la matriz que traduce “qué tan rápido giran los motores” "
               "a “qué tan rápido se mueve la mano”.")
    if singular:
        dice(f"Esta postura es <b>singular</b>: el rango bajó a {rango} de {completo}, así que hay "
             f"{completo - rango} dirección(es) en las que la mano <b>no se puede mover</b>, "
             f"por más que giren los motores.")
        st.warning("Cerca de una singularidad las velocidades que pide el robot se van al cielo. "
                   "En la práctica se evita esa zona, o se usa amortiguamiento para que las "
                   "velocidades se queden finitas a cambio de un poquito de error.", icon="⚠️")
    else:
        extra = ("Aunque la manipulabilidad está muy baja: estás cerquita de una singularidad."
                 if w < 0.01 else "La postura está sana, lejos de singularidades.")
        dice(f"Con sus {n} articulación(es), el robot controla las <b>{completo}</b> direcciones "
             f"que le tocan. {extra}")
    pista("<b>v = J·q̇</b>. Cada columna de J dice cómo mueve la mano una articulación cuando gira a "
          "1 rad/s. Los tres primeros renglones son la velocidad de traslación y los tres últimos, la de giro. "
          "La misma matriz transpuesta relaciona la fuerza en la mano con los pares de los motores.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rango", f"{rango} de {completo}", help="Cuántas direcciones puede controlar aquí.")
    c2.metric("Manipulabilidad", n_fmt(w, 5), help="Vale exactamente cero en una singularidad.")
    c3.metric("Nº de condición", "∞" if sv[-1] < 1e-12 else n_fmt(sv[0] / sv[-1], 1),
              help="Qué tan desequilibrado está: si es enorme, hay direcciones casi imposibles.")

    etiquetas = ["velocidad en x", "velocidad en y", "velocidad en z",
                 "giro sobre x", "giro sobre y", "giro sobre z"]
    dfj = pd.DataFrame(np.round(J, 4), index=etiquetas,
                       columns=[f"eje {i + 1}" for i in range(n)])
    st.dataframe(dfj, **(argumentos_tabla or {}))
    with st.expander("Ver el jacobiano como matriz y sus valores singulares"):
        st.latex(latex_matriz(J, "J", 3))
        st.markdown("Valores singulares: " + "  ·  ".join(n_fmt(v, 4) for v in sv))

    st.divider()
    st.markdown("#### Si muevo los motores así, ¿cómo se mueve la mano?")
    qd = vector_entrada([f"q̇{i + 1}" for i in range(n)], [1.0] + [0.0] * (n - 1), "qdj", 0.1)
    v6 = J @ qd
    c1, c2 = st.columns(2)
    c1.metric("La mano se traslada a", f"{np.linalg.norm(v6[:3]):.3f} m/s")
    c2.metric("y gira a", f"{np.linalg.norm(v6[3:]):.3f} rad/s")
    tabla(pd.DataFrame({"Componente": etiquetas, "Valor": [n_fmt(v, 4) for v in v6],
                        "Unidad": ["m/s"] * 3 + ["rad/s"] * 3}))

    st.divider()
    st.markdown("#### Quiero que la mano se mueva así, ¿qué velocidades le pido a los motores?")
    dest = vector_entrada(["vx [m/s]", "vy [m/s]", "vz [m/s]",
                           "ωx [rad/s]", "ωy [rad/s]", "ωz [rad/s]"],
                          [0.1, 0, 0, 0, 0, 0], "tw", 0.05)
    lam = st.slider("Amortiguamiento λ", 0.0, 0.3, 0.01, 0.005,
                    help="Sirve de seguro cerca de las singularidades: sube λ si salen velocidades absurdas.")
    A = J @ J.T + (lam ** 2) * np.eye(6)
    try:
        qd_nec = J.T @ np.linalg.solve(A, dest)
        err = float(np.linalg.norm(dest - J @ qd_nec))
        k = int(np.argmax(np.abs(qd_nec)))
        dice(f"El que más rápido tendría que ir es el <b>eje {k + 1}</b>, a "
             f"{n_fmt(abs(qd_nec[k]))} {'rad/s' if rob.eslabones[k].es_rotacion else 'm/s'}. "
             + (f"Por el amortiguamiento queda un error de {n_fmt(err, 4)} en el movimiento."
                if err > 1e-3 else "El movimiento pedido se consigue exacto."))
        tabla(pd.DataFrame({"Eje": [f"{i + 1}" for i in range(n)],
                            "Velocidad que necesita": [n_fmt(v, 4) for v in qd_nec],
                            "Unidad": ["rad/s" if e.es_rotacion else "m/s" for e in rob.eslabones]}))
    except np.linalg.LinAlgError:
        st.warning("No se pudo resolver: sube el amortiguamiento λ.", icon="⚠️")

    st.divider()
    st.markdown("#### Si la mano carga o empuja algo, ¿qué aguanta cada motor?")
    h = vector_entrada(["Fx [N]", "Fy [N]", "Fz [N]", "Mx [N·m]", "My [N·m]", "Mz [N·m]"],
                       [0, 0, -50, 0, 0, 0], "wr", 5.0)
    tau = J.T @ h
    k = int(np.argmax(np.abs(tau)))
    dice(f"Para aguantar eso sin moverse, el más exigido es el <b>eje {k + 1}</b> con "
         f"<b>{n_fmt(abs(tau[k]), 2)} {unidad_par(rob.eslabones[k])}</b>. "
         f"Esto es solo por la carga: el peso del propio brazo se calcula en la pestaña de motores.")
    tabla(pd.DataFrame({"Eje": [f"{i + 1}" for i in range(n)],
                        "τ = Jᵀ·F": [n_fmt(v, 3) for v in tau],
                        "Unidad": [unidad_par(e) for e in rob.eslabones]}))
