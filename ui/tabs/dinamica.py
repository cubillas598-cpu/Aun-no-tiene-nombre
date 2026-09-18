"""Pestana de dinamica y dimensionamiento de actuadores."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from domain.inertia import FORMAS, steiner
from domain.robot import Robot
from ui.components import dice, encabezado, pista, tabla
from ui.state import latex_matriz, n_fmt, ss, unidad_par


def tab_dinamica(vector_entrada, argumentos_tabla: dict | None = None) -> None:
    rob: Robot = ss().robot
    q = ss().q
    n = rob.n
    encabezado("¿De cuánto tienen que ser los motores?",
               "Esto ya no es geometría: aquí entran las masas, las inercias y la gravedad.")
    pista("Se resuelve con el método recursivo de <b>Newton–Euler</b> y el resultado se parte en los "
          "pedazos del modelo clásico:<br><b>τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + fricción + carga</b><br>"
          "inercia, efectos de Coriolis y centrífugos, peso, fricción y lo que lleve la mano. "
          "Así se ve de dónde sale cada N·m.")
    st.markdown("**¿A qué velocidad y con qué aceleración se está moviendo?**")
    qd = vector_entrada([f"q̇{i + 1}" for i in range(n)], [0.0] * n, "qdd_v", 0.1)
    qdd = vector_entrada([f"q̈{i + 1}" for i in range(n)], [0.0] * n, "qdd_a", 0.1)
    with st.expander("¿La mano está cargando o empujando algo?"):
        carga = vector_entrada(["Fx [N]", "Fy [N]", "Fz [N]", "Mx [N·m]", "My [N·m]", "Mz [N·m]"],
                               [0.0] * 6, "cg", 1.0)

    traza: dict = {}
    tau = rob.pares(q, qd, qdd, carga=carga, traza=traza)
    M = rob.matriz_inercia(q)
    gv = rob.vector_gravedad(q)
    cv = rob.vector_coriolis(q, qd)
    fr = np.array([e.friccion_viscosa * qd[i] + e.friccion_seca * math.copysign(1.0, qd[i]) * (qd[i] != 0)
                   for i, e in enumerate(rob.eslabones)])
    inerc = M @ qdd
    ext = tau - (inerc + cv + gv + fr)
    en = rob.energia(q, qd, M)

    k = int(np.argmax(np.abs(tau)))
    trozos = {"la inercia": abs(inerc[k]), "la gravedad": abs(gv[k]), "Coriolis": abs(cv[k]),
              "la fricción": abs(fr[k]), "la carga de la mano": abs(ext[k])}
    total = sum(trozos.values()) or 1.0
    mayor = max(trozos, key=trozos.get)
    gira = rob.eslabones[k].es_rotacion
    dice(f"El actuador más exigido es el del <b>eje {k + 1}</b>: necesita "
         f"<b>{n_fmt(abs(tau[k]), 2)} {unidad_par(rob.eslabones[k])}</b> de "
         f"{'par' if gira else 'fuerza'}. De esa cantidad, la mayor parte viene de "
         f"<b>{mayor}</b> ({trozos[mayor] / total * 100:.0f} %). Al escoger el "
         f"{'motor y su reductor' if gira else 'actuador lineal'}, deja además un margen de seguridad.")

    cols = st.columns(min(n, 6))
    for i in range(min(n, 6)):
        cols[i].metric(f"Eje {i + 1}", f"{tau[i]:.2f}", unidad_par(rob.eslabones[i]))
    st.markdown("**¿De dónde sale cada N·m?**")
    desglose = pd.DataFrame({
        "De dónde viene": ["Inercia  M(q)q̈", "Coriolis y centrífugo  C(q,q̇)q̇", "Peso  g(q)",
                           "Fricción", "Carga en la mano", "TOTAL  τ"],
        **{f"eje {i + 1}": [n_fmt(inerc[i], 3), n_fmt(cv[i], 3), n_fmt(gv[i], 3),
                            n_fmt(fr[i], 3), n_fmt(ext[i], 3), n_fmt(tau[i], 3)] for i in range(n)}})
    tabla(desglose)

    c1, c2, c3 = st.columns(3)
    c1.metric("Energía cinética", f"{en['cinetica']:.3f} J")
    c2.metric("Energía potencial", f"{en['potencial']:.3f} J")
    c3.metric("Energía total", f"{en['total']:.3f} J")

    with st.expander("Ver las matrices M(q) y C(q,q̇)"):
        st.latex(latex_matriz(M, "M(q)", 4))
        st.latex(latex_matriz(rob.matriz_coriolis(q, qd), "C(q,\\dot q)", 4))
        st.markdown("La **diagonal de M** es la inercia que siente cada motor al mover solo su eje; lo de "
                    "fuera de la diagonal es el acoplamiento, o sea cuánto se estorban entre ellos. "
                    "M siempre sale simétrica y definida positiva: si no, hay un dato mal capturado.")

    with st.expander("Ver la recursión paso a paso (lo que pide la tarea)"):
        st.markdown("**Primero de la base a la mano**, propagando el movimiento:")
        tabla(pd.DataFrame({
            "Eje": [f"{i + 1}" for i in range(n)],
            "ω (giro)": [", ".join(n_fmt(v, 2) for v in traza["omega"][i]) for i in range(n)],
            "ω̇ (aceleración de giro)": [", ".join(n_fmt(v, 2) for v in traza["omega_punto"][i]) for i in range(n)],
            "a del origen": [", ".join(n_fmt(v, 2) for v in traza["a_origen"][i]) for i in range(n)],
            "a del centro de masa": [", ".join(n_fmt(v, 2) for v in traza["a_centro_masa"][i]) for i in range(n)]}))
        st.markdown("**Y luego de la mano a la base**, propagando fuerzas:")
        tabla(pd.DataFrame({
            "Eje": [f"{i + 1}" for i in range(n)],
            "fuerza f [N]": [", ".join(n_fmt(v, 2) for v in traza["fuerza"][i]) for i in range(n)],
            "momento μ [N·m]": [", ".join(n_fmt(v, 2) for v in traza["momento"][i]) for i in range(n)],
            "τ del actuador": [n_fmt(traza["tau"][i], 3) for i in range(n)]}))

    with st.expander("Calculadora de tensores de inercia (para llenar la tabla de masas)"):
        forma = st.selectbox("Forma de la pieza", list(FORMAS))
        funcion, nombres = FORMAS[forma]
        masa = st.number_input("masa [kg]", value=1.0, step=0.1, key="forma_m")
        dims = vector_entrada(nombres, [0.1] * len(nombres), "dim", 0.01)
        I = funcion(masa, *dims)
        st.latex(latex_matriz(I, "I_{cm}", 6))
        d = vector_entrada(["dx [m]", "dy [m]", "dz [m]"], [0.0] * 3, "steiner", 0.01)
        if np.linalg.norm(d) > 0:
            st.markdown("Trasladado a ese punto con el teorema de los ejes paralelos:")
            st.latex(latex_matriz(steiner(I, masa, d), "I", 6))
        st.caption("Copia los números de la diagonal a las columnas Ixx, Iyy, Izz del eslabón "
                   "correspondiente, en la pestaña Editar mi robot.")
