"""Pestana de trayectorias articulares y movimiento de la mano."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from domain.robot import Robot
from domain.rotations import rad
from domain.trajectories import cubica, quintica, trapezoidal
from ui.components import dice, encabezado, pista
from ui.figures import figura_curvas, figura_robot
from ui.state import a_interno, a_pantalla, ss, uni, unidad_junta, unidad_par


def tab_trayectorias(vector_entrada, argumentos_grafica: dict | None = None) -> None:
    rob: Robot = ss().robot
    q = ss().q
    n = rob.n
    encabezado("¿Cómo lo hago moverse suave de un lado a otro?",
               "Una trayectoria es la receta de cómo cambia cada articulación con el tiempo.")
    pista("Los <b>polinomios</b> reparten el movimiento en una curva suave: el cúbico garantiza velocidad "
          "continua y el quíntico además arranca y frena con aceleración cero, lo más amable para la "
          "mecánica. El <b>trapezoidal</b> acelera, viaja a velocidad constante y frena: es el que usan "
          "casi todos los robots industriales porque aprovecha mejor el motor.")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    perfil = c1.selectbox("Perfil", ["Polinomio cúbico", "Polinomio quíntico", "Trapezoidal de velocidad"])
    tf = c2.number_input("¿Cuánto debe durar? [s]", value=2.0, min_value=0.1, step=0.5)
    acel = c3.number_input(f"Aceleración [{uni()}/s² o m/s²]", value=180.0 if ss().grados else 4.0,
                           step=10.0, disabled=not perfil.startswith("Trapez"),
                           help="Solo se usa en el perfil trapezoidal.")

    st.markdown("**¿De qué postura a qué postura?**")
    base = pd.DataFrame({
        "Eje": [f"{i + 1}" for i in range(n)],
        "Empieza en": [round(a_pantalla(q[i]) if e.es_rotacion else q[i], 3)
                       for i, e in enumerate(rob.eslabones)],
        "Termina en": [round(a_pantalla(np.clip(q[i] + (rad(30) if e.es_rotacion else 0.1),
                                            e.qmin, e.qmax)) if e.es_rotacion else
                       np.clip(q[i] + 0.1, e.qmin, e.qmax), 3)
                       for i, e in enumerate(rob.eslabones)],
        "Unidad": [unidad_junta(e) for e in rob.eslabones]})
    edit = st.data_editor(base, hide_index=True,
                          disabled=["Eje", "Unidad"], key=f"tray_{ss().version}_{int(ss().grados)}",
                          column_config={"Empieza en": st.column_config.NumberColumn(format="%.3f"),
                                         "Termina en": st.column_config.NumberColumn(format="%.3f")})

    funcs = []
    for i, e in enumerate(rob.eslabones):
        conv = a_interno if e.es_rotacion else (lambda v: float(v))
        q0 = conv(float(edit["Empieza en"][i]))
        qf = conv(float(edit["Termina en"][i]))
        if not (e.qmin <= q0 <= e.qmax and e.qmin <= qf <= e.qmax):
            st.error(f"El eje {i + 1} tiene una postura fuera de sus límites mecánicos.", icon="⚠️")
            return
        if perfil.startswith("Polinomio cúbico"):
            funcs.append(cubica(q0, qf, 0.0, 0.0, tf))
        elif perfil.startswith("Polinomio quíntico"):
            funcs.append(quintica(q0, qf, 0.0, 0.0, 0.0, 0.0, tf))
        else:
            funcs.append(trapezoidal(q0, qf, tf, conv(acel)))

    N = 150
    t = np.linspace(0, tf, N)
    muestras = np.array([[f(ti) for f in funcs] for ti in t])
    Q, QD, QDD = muestras[:, :, 0], muestras[:, :, 1], muestras[:, :, 2]
    idx = np.arange(0, N, 3)
    t_tau = t[idx]
    TAU = np.array([rob.pares(Q[k], QD[k], QDD[k]) for k in idx])
    camino = np.array([rob.posicion(Q[k]) for k in range(0, N, 2)])

    pico = np.max(np.abs(TAU), axis=0)
    ip = int(np.argmax(pico))
    tp = float(t_tau[int(np.argmax(np.abs(TAU[:, ip])))])
    dice(f"Durante el movimiento el que más trabaja es el <b>eje {ip + 1}</b>: llega a "
         f"<b>{pico[ip]:.2f} {unidad_par(rob.eslabones[ip])}</b> en el segundo {tp:.2f}. "
         f"Si le das más tiempo al movimiento los pares bajan rapidísimo, porque van con el "
         f"cuadrado del tiempo: al doble de duración, la cuarta parte de aceleración.")
    if perfil.startswith("Trapez") and any(getattr(f, "insuficiente", False) for f in funcs):
        st.warning("La aceleración que pediste no alcanza para hacerlo en ese tiempo, así que se subió "
                   "al mínimo posible. El trapezoidal necesita al menos 4·Δq/t².", icon="⚠️")

    cols = st.columns(min(n, 6))
    for i in range(min(n, 6)):
        cols[i].metric(f"Pico eje {i + 1}", f"{pico[i]:.2f}", unidad_par(rob.eslabones[i]))

    esc = a_pantalla if all(e.es_rotacion for e in rob.eslabones) else (lambda v: v)
    st.plotly_chart(figura_curvas([
        (f"Posición de cada articulación [{uni()} o m]", t, np.vectorize(esc)(Q)),
        (f"Velocidad [{uni()}/s o m/s]", t, np.vectorize(esc)(QD)),
        (f"Aceleración [{uni()}/s² o m/s²]", t, np.vectorize(esc)(QDD)),
        ("Par que pide cada motor [N·m o N]", t_tau, TAU)], rob),
        config={"displaylogo": False}, **(argumentos_grafica or {}))

    st.divider()
    st.markdown("#### Míralo moverse")
    inst = st.slider("Instante del movimiento [s]", 0.0, float(tf), 0.0, float(tf) / 60,
                     help="Deslízalo para ver el robot en cada momento del recorrido.")
    q_inst = np.array([f(inst)[0] for f in funcs])
    fig = figura_robot(rob, q_inst, altura=430)
    fig.add_trace(go.Scatter3d(x=camino[:, 0], y=camino[:, 1], z=camino[:, 2], mode="lines",
                               name="camino de la mano", line=dict(color="#E0862D", width=5),
                               hoverinfo="skip"))
    st.plotly_chart(fig, config={"displaylogo": False}, **(argumentos_grafica or {}))
    if st.button("Dejar el robot en esta postura"):
        ss().q = q_inst
        ss().version += 1
        st.rerun()
