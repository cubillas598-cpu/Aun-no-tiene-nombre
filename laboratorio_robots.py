"""
================================================================================
  LABORATORIO DE CINEMATICA Y DINAMICA DE ROBOTS
================================================================================

  Como usarlo (una sola vez):

      pip install streamlit numpy plotly pandas

  Y despues, cada vez que quieras abrirlo:

      streamlit run laboratorio_robots.py

  Se abre solo en tu navegador, en la direccion http://localhost:8501
  No necesitas saber programar para usarlo.

  Si algo no funciona, escribe:

      python laboratorio_robots.py --revisar

  y te dira exactamente que falta.
  Convencion: Denavit-Hartenberg estandar. Unidades del SI (m, kg, N.m, s).
================================================================================
"""

from __future__ import annotations

import math
import os
import re
import sys
import hashlib
from typing import Sequence

import numpy as np

# ==============================================================================
#  PARTE 1 - MATEMATICAS
#  Las operaciones puras viven en domain/ para poder probarlas sin Streamlit.
# ==============================================================================

from domain.rotations import (R_desde_cuaternion, R_desde_eje_angulo, R_desde_rpy,
                               R_desde_zyz, cuaternion_desde_R, deg,
                               eje_angulo_desde_R, homogenea, inversa_homogenea,
                               rad, rot_x, rot_y, rot_z, rpy_desde_R, zyz_desde_R)


# El dominio usa el modelo desacoplado; este import mantiene estable la API de la app.
from domain.robot import Eslabon, Robot
from domain.catalog import catalogo
from domain.inverse import inversa_2R, inversa_3R_plano, inversa_antropomorfico
from domain.trajectories import cubica, quintica, trapezoidal
from domain.inertia import (FORMAS, inercia_cilindro, inercia_esfera, inercia_prisma,
                             inercia_varilla, steiner)
from ui.state import (a_interno, a_pantalla, iniciar_estado, latex_matriz, n_fmt, ss,
                      uni, unidad_junta, unidad_par, _cargar)
from ui.figures import figura_curvas, figura_robot
from ui.components import dice, encabezado, pista, tabla
from ui.tabs.empezar import tab_empezar
from ui.tabs.directa import tab_directa
from ui.tabs.inversa import tab_inversa as tab_inversa_modular


# ==============================================================================
#  PARTE 2 - INTERFAZ
#  Streamlit para la pantalla, Plotly para el robot en 3D.
# ==============================================================================

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- compatibilidad entre versiones de Streamlit -----------------------------
# Streamlit cambio la forma de decir "ocupa todo el ancho": antes era
# use_container_width=True y ahora es width="stretch", y no todas las funciones
# cambiaron a la vez. Aqui se detecta que entiende la version instalada, para que
# el programa funcione igual con una version vieja o con la mas nueva.
import inspect
import logging

# Al abrirlo con "python" en vez de "streamlit run", Streamlit suelta un aviso tecnico
# que no significa nada para quien lo usa. Lo callamos.
for _ruido in ("streamlit.runtime.scriptrunner_utils.script_run_context",
               "streamlit.runtime.scriptrunner.script_run_context"):
    logging.getLogger(_ruido).setLevel(logging.ERROR)

VERSION_MINIMA = (1, 28)
_VERSION = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
_API_NUEVA = "width" in inspect.signature(st.button).parameters


def _ancho(funcion) -> dict:
    """Devuelve el argumento que esta version de Streamlit entiende."""
    try:
        parametros = inspect.signature(funcion).parameters
    except (TypeError, ValueError):
        return {}
    if _API_NUEVA and "width" in parametros:
        return {"width": "stretch"}
    if "use_container_width" in parametros:
        return {"use_container_width": True}
    return {}


A_BOTON = _ancho(st.button)
A_TABLA = _ancho(st.dataframe)
A_EDITOR = _ancho(st.data_editor)
A_GRAFICA = _ancho(st.plotly_chart)
A_DESCARGA = _ancho(st.download_button)

TINTA = "#171A1F"
SUAVE = "#6B7580"
MARINO = "#1B3A6B"
AMBAR = "#E0862D"
VERDE = "#177A4C"
ROJO = "#C0392B"
COLOR_X, COLOR_Y, COLOR_Z = "#E0544A", "#2F9E63", "#2C6BD6"
PALETA = ["#1B3A6B", "#E0862D", "#177A4C", "#8E44AD", "#C0392B", "#0E7C86", "#B7950B", "#5D6D7E"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', system-ui, sans-serif !important; letter-spacing: -0.02em; }
[data-testid="stAppViewContainer"] { background: #F6F6F3; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1500px; }

[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E6E5E0; }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* tarjetas de resultado */
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E6E5E0; border-radius: 14px;
  padding: 14px 16px 10px; box-shadow: 0 1px 2px rgba(23,26,31,.04);
}
[data-testid="stMetricLabel"] { color: #6B7580; font-size: .8rem; font-weight: 500; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; color: #171A1F; }

/* pestañas tipo pastilla */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E6E5E0; padding-bottom: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
  background: #FFFFFF; border: 1px solid #E6E5E0; border-radius: 999px;
  padding: 7px 16px; font-size: .92rem; font-weight: 500; color: #6B7580;
}
.stTabs [aria-selected="true"] { background: #1B3A6B !important; color: #FFFFFF !important; border-color: #1B3A6B !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* bloques propios */
.tarjeta { background:#FFF; border:1px solid #E6E5E0; border-radius:16px; padding:18px 20px; margin-bottom:14px;
           box-shadow:0 1px 2px rgba(23,26,31,.04); }
.dice { font-size:1.02rem; line-height:1.6; border-left:3px solid #1B3A6B; padding:2px 0 2px 14px; margin:10px 0 16px; }
.dice b { color:#1B3A6B; }
.pista { background:#FFF; border:1px solid #E6E5E0; border-left:3px solid #E0862D; border-radius:12px;
         padding:12px 16px; margin:12px 0; font-size:.93rem; color:#3C4450; }
.pista b { color:#171A1F; }
.paso { display:flex; gap:14px; align-items:flex-start; margin-bottom:16px; }
.paso .num { flex:0 0 32px; height:32px; border-radius:50%; border:1.5px solid #1B3A6B; color:#1B3A6B;
             display:grid; place-items:center; font-family:'JetBrains Mono',monospace; font-size:.9rem; }
.paso .txt b { display:block; font-family:'Space Grotesk',sans-serif; font-size:1.02rem; margin-bottom:2px; }
.chip { display:inline-block; background:#EDF1F7; color:#1B3A6B; border-radius:999px;
        padding:3px 11px; font-size:.8rem; font-weight:600; margin-right:6px; }
.mini { color:#6B7580; font-size:.86rem; }
code, .stCode, pre { font-family:'JetBrains Mono', monospace !important; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #E6E5E0; }
div.stButton > button {
  border-radius: 10px; border:1px solid #E6E5E0; background:#FFF; color:#171A1F; font-weight:500; padding:.45rem 1rem;
}
div.stButton > button:hover { border-color:#1B3A6B; color:#1B3A6B; }
div.stButton > button[kind="primary"] { background:#1B3A6B; color:#FFF; border-color:#1B3A6B; }
</style>
"""

DIAGRAMA_DH = """
<svg viewBox="0 0 520 250" style="width:100%;max-width:520px;background:#fff;border:1px solid #E6E5E0;border-radius:12px">
  <defs><marker id="fl" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#6B7580"/></marker></defs>
  <line x1="110" y1="228" x2="110" y2="28" stroke="#2C6BD6" stroke-width="2"/>
  <text x="117" y="40" font-size="12" fill="#2C6BD6" font-family="monospace">z(i-1)</text>
  <line x1="390" y1="210" x2="428" y2="58" stroke="#2C6BD6" stroke-width="2"/>
  <text x="396" y="52" font-size="12" fill="#2C6BD6" font-family="monospace">z(i)</text>
  <line x1="110" y1="196" x2="205" y2="196" stroke="#E0544A" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="152" y="211" font-size="11" fill="#E0544A" font-family="monospace">x(i-1)</text>
  <line x1="110" y1="118" x2="392" y2="118" stroke="#E0544A" stroke-width="2" marker-end="url(#fl)"/>
  <text x="352" y="110" font-size="12" fill="#E0544A" font-family="monospace">x(i)</text>
  <circle cx="110" cy="196" r="4" fill="#171A1F"/><text x="72" y="200" font-size="11" fill="#171A1F" font-family="monospace">O(i-1)</text>
  <circle cx="390" cy="118" r="4" fill="#171A1F"/><text x="397" y="136" font-size="11" fill="#171A1F" font-family="monospace">O(i)</text>
  <line x1="88" y1="196" x2="88" y2="118" stroke="#6B7580" stroke-width="1" marker-end="url(#fl)"/>
  <text x="58" y="161" font-size="14" fill="#171A1F" font-family="monospace">d</text>
  <line x1="120" y1="101" x2="380" y2="101" stroke="#6B7580" stroke-width="1" marker-end="url(#fl)"/>
  <text x="248" y="94" font-size="14" fill="#171A1F" font-family="monospace">a</text>
  <path d="M148,196 A38,38 0 0,0 133,169" fill="none" stroke="#6B7580" stroke-width="1.4" marker-end="url(#fl)"/>
  <text x="153" y="180" font-size="14" fill="#171A1F" font-family="monospace">&#952;</text>
  <path d="M390,84 A34,34 0 0,1 412,90" fill="none" stroke="#6B7580" stroke-width="1.4" marker-end="url(#fl)"/>
  <text x="419" y="84" font-size="14" fill="#171A1F" font-family="monospace">&#945;</text>
  <text x="16" y="243" font-size="11.5" fill="#6B7580">Giras &#952; y avanzas d sobre el eje viejo; luego avanzas a y giras &#945; sobre el eje nuevo.</text>
</svg>
"""


def poner_q(q_nuevo: Sequence[float]) -> None:
    """Cambia la postura y fuerza a que los deslizadores se redibujen."""
    ss().q = np.asarray(q_nuevo, float)
    ss().version += 1


def vector_entrada(etiquetas: Sequence[str], valores: Sequence[float], prefijo: str,
                   paso: float = 0.1, formato: str = "%.3f") -> np.ndarray:
    """Fila de casillas numericas, una por componente."""
    cols = st.columns(len(etiquetas))
    salida = []
    for col, et, v in zip(cols, etiquetas, valores):
        with col:
            salida.append(st.number_input(et, value=float(v), step=paso, format=formato,
                                          key=f"{prefijo}_{et}"))
    return np.array(salida, float)


# ------------------------------------------------------------- barra lateral
def barra_lateral() -> None:
    with st.sidebar:
        st.markdown("## 🦾 Mi robot")
        nombres = list(catalogo())
        elegido = st.selectbox("Elige un robot para empezar", nombres, key="sel_robot",
                               index=nombres.index(ss().nombre) if ss().nombre in nombres else 0)
        if elegido != ss().get("_elegido"):
            _cargar(elegido)

        rob: Robot = ss().robot
        st.caption(f"{rob.n} articulaciones · "
                   f"{sum(1 for e in rob.eslabones if e.es_rotacion)} giran, "
                   f"{sum(1 for e in rob.eslabones if not e.es_rotacion)} deslizan")

        st.divider()
        st.markdown("#### Mueve las articulaciones")
        q = np.asarray(ss().q, float).copy()
        for i, e in enumerate(rob.eslabones):
            clave = f"q_{ss().version}_{int(ss().grados)}_{i}"
            if e.es_rotacion:
                lo, hi = a_pantalla(e.qmin), a_pantalla(e.qmax)
                paso = 1.0 if ss().grados else 0.02
                v = st.slider(f"Eje {i + 1} · gira   [{uni()}]", float(lo), float(hi),
                              float(np.clip(a_pantalla(q[i]), lo, hi)), paso, key=clave)
                q[i] = a_interno(v)
            else:
                v = st.slider(f"Eje {i + 1} · desliza   [m]", float(e.qmin), float(e.qmax),
                              float(np.clip(q[i], e.qmin, e.qmax)), 0.005, key=clave)
                q[i] = v
        ss().q = q

        c1, c2 = st.columns(2)
        if c1.button("Todo en cero", **A_BOTON):
            lims = rob.limites()
            poner_q(np.clip(np.zeros(rob.n), lims[:, 0], lims[:, 1]))
            st.rerun()
        if c2.button("Postura inicial", **A_BOTON):
            _, q0 = catalogo()[ss().nombre]() if ss().nombre in catalogo() else (None, ss().q)
            poner_q(q0)
            st.rerun()

        if st.checkbox("Ver hasta dónde alcanza", key="ver_alcance",
                       help="Sortea miles de posturas y dibuja todos los puntos que la mano puede tocar."):
            if ss().nube is None or len(ss().nube) == 0:
                ss().nube = rob.espacio_trabajo(3000)
        else:
            ss().nube = None

        st.divider()
        st.radio("Los ángulos los quiero en", ["grados", "radianes"],
                 key="unidad_ang", horizontal=True)
        st.toggle("Mostrar explicaciones", key="ver_explicaciones",
                  help="Apágalo cuando ya solo quieras los números.")
        ss().grados = ss().unidad_ang == "grados"
        ss().explicar = bool(ss().ver_explicaciones)
        st.caption("Hecho con Python · Streamlit · Plotly")


# -------------------------------------------------------------- encabezado
def encabezado_principal() -> None:
    rob: Robot = ss().robot
    q = ss().q
    st.markdown("# Laboratorio de robots")
    st.markdown('<p class="mini">Cinemática y dinámica de robots, paso a paso. '
                'Mueve las articulaciones en la barra de la izquierda y mira qué pasa.</p>',
                unsafe_allow_html=True)

    izq, der = st.columns([2.1, 1], gap="large")
    with izq:
        st.plotly_chart(figura_robot(rob, q, ss().nube), **A_GRAFICA,
                        config={"displaylogo": False,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
        st.markdown('<p class="mini">Arrastra para girar el robot · rueda del ratón para acercarte · '
                    'las barras azules son articulaciones que giran y las naranjas, que deslizan.</p>',
                    unsafe_allow_html=True)
    with der:
        T = rob.pose(q)
        p = T[:3, 3]
        eje, ang = eje_angulo_desde_R(T[:3, :3])
        roll, pitch, yaw = rpy_desde_R(T[:3, :3])
        st.markdown("#### ¿Dónde quedó la mano?")
        c1, c2, c3 = st.columns(3)
        c1.metric("x", f"{p[0]:.3f} m")
        c2.metric("y", f"{p[1]:.3f} m")
        c3.metric("z", f"{p[2]:.3f} m")
        st.metric("Distancia al origen", f"{np.linalg.norm(p):.3f} m")
        st.markdown(
            f'<div class="tarjeta"><span class="chip">orientación</span><br><br>'
            f'<span class="mini">Girada <b>{n_fmt(a_pantalla(ang), 1)} {uni()}</b> '
            f'alrededor del eje ({", ".join(n_fmt(v, 2) for v in eje)})</span><br><br>'
            f'<span class="mini">roll {n_fmt(a_pantalla(roll), 1)} · '
            f'pitch {n_fmt(a_pantalla(pitch), 1)} · yaw {n_fmt(a_pantalla(yaw), 1)} {uni()}</span></div>',
            unsafe_allow_html=True)
        tope = [i + 1 for i, e in enumerate(rob.eslabones)
                if q[i] <= e.qmin + 1e-9 or q[i] >= e.qmax - 1e-9]
        if tope:
            st.warning(f"El eje {', '.join(map(str, tope))} está en su tope mecánico: "
                       f"no puede seguir en esa dirección.", icon="⚠️")


# ============================ PESTAÑA: JACOBIANO ============================
def tab_jacobiano() -> None:
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
    st.dataframe(dfj, **A_TABLA)
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


# ============================= PESTAÑA: DINAMICA =============================
def tab_dinamica() -> None:
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


# =========================== PESTAÑA: TRAYECTORIAS ===========================
def tab_trayectorias() -> None:
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
    edit = st.data_editor(base, hide_index=True, **A_EDITOR,
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
    muestras = np.array([[f(ti) for f in funcs] for ti in t])     # (N, n, 3)
    Q, QD, QDD = muestras[:, :, 0], muestras[:, :, 1], muestras[:, :, 2]
    paso = 3                                                       # los pares cuestan: malla mas gruesa
    idx = np.arange(0, N, paso)
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
        config={"displaylogo": False}, **A_GRAFICA)

    st.divider()
    st.markdown("#### Míralo moverse")
    inst = st.slider("Instante del movimiento [s]", 0.0, float(tf), 0.0, float(tf) / 60,
                     help="Deslízalo para ver el robot en cada momento del recorrido.")
    q_inst = np.array([f(inst)[0] for f in funcs])
    fig = figura_robot(rob, q_inst, altura=430)
    fig.add_trace(go.Scatter3d(x=camino[:, 0], y=camino[:, 1], z=camino[:, 2], mode="lines",
                               name="camino de la mano",
                               line=dict(color=AMBAR, width=5), hoverinfo="skip"))
    st.plotly_chart(fig, config={"displaylogo": False}, **A_GRAFICA)
    if st.button("Dejar el robot en esta postura"):
        poner_q(q_inst)
        st.rerun()


# ============================ PESTAÑA: ROTACIONES ============================
def tab_rotaciones() -> None:
    encabezado("Convertir entre todas las formas de escribir una orientación",
               "La misma orientación se puede escribir de cinco maneras. Aquí pasas de una a otra.")
    pista("Los <b>ángulos</b> son fáciles de leer pero tienen posturas donde se pierde información "
          "(el famoso bloqueo de cardán). El <b>cuaternión</b> no tiene ese problema y por eso es el que "
          "se usa para interpolar giros.")

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
        num_rows="dynamic", hide_index=True, key="comp_pasos", **A_EDITOR,
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


# =========================== PESTAÑA: EDITAR ROBOT ===========================
def _df_dh(rob: Robot) -> pd.DataFrame:
    return pd.DataFrame({
        "Tipo": ["gira" if e.es_rotacion else "desliza" for e in rob.eslabones],
        f"θ [{uni()}]": [round(a_pantalla(e.theta), 4) for e in rob.eslabones],
        "d [m]": [round(e.d, 4) for e in rob.eslabones],
        "a [m]": [round(e.a, 4) for e in rob.eslabones],
        f"α [{uni()}]": [round(a_pantalla(e.alpha), 4) for e in rob.eslabones],
        "q mínima": [round(a_pantalla(e.qmin) if e.es_rotacion else e.qmin, 3) for e in rob.eslabones],
        "q máxima": [round(a_pantalla(e.qmax) if e.es_rotacion else e.qmax, 3) for e in rob.eslabones]})


def _df_masas(rob: Robot) -> pd.DataFrame:
    return pd.DataFrame({
        "masa [kg]": [round(e.masa, 5) for e in rob.eslabones],
        "cm x": [round(float(e.centro_masa[0]), 5) for e in rob.eslabones],
        "cm y": [round(float(e.centro_masa[1]), 5) for e in rob.eslabones],
        "cm z": [round(float(e.centro_masa[2]), 5) for e in rob.eslabones],
        "Ixx": [round(float(e.inercia[0, 0]), 6) for e in rob.eslabones],
        "Iyy": [round(float(e.inercia[1, 1]), 6) for e in rob.eslabones],
        "Izz": [round(float(e.inercia[2, 2]), 6) for e in rob.eslabones],
        "fricción viscosa": [round(e.friccion_viscosa, 6) for e in rob.eslabones],
        "fricción seca": [round(e.friccion_seca, 6) for e in rob.eslabones],
        "inercia del motor": [round(e.inercia_motor, 6) for e in rob.eslabones]})


def _datos_robot(rob: Robot) -> dict:
    return {"nombre": rob.nombre, "gravedad": np.asarray(rob.gravedad).tolist(),
            "herramienta": None if rob.herramienta is None else np.asarray(rob.herramienta).tolist(),
            "eslabones": [{"tipo": e.tipo, "theta": e.theta, "d": e.d, "a": e.a, "alpha": e.alpha,
                           "qmin": e.qmin, "qmax": e.qmax, "masa": e.masa,
                           "centro_masa": np.asarray(e.centro_masa).tolist(),
                           "inercia": np.asarray(e.inercia).tolist(),
                           "friccion_viscosa": e.friccion_viscosa,
                           "friccion_seca": e.friccion_seca,
                           "inercia_motor": e.inercia_motor} for e in rob.eslabones]}


def _robot_desde_datos(datos: dict) -> Robot:
    if not isinstance(datos, dict):
        raise ValueError("El archivo debe contener un objeto JSON.")
    eslabones = datos.get("eslabones")
    if not isinstance(eslabones, list) or not eslabones or len(eslabones) > 50:
        raise ValueError("El archivo debe contener entre 1 y 50 eslabones.")
    try:
        robot = Robot(
            str(datos.get("nombre", "Mi robot")),
            [Eslabon(tipo=e["tipo"], theta=float(e["theta"]), d=float(e["d"]), a=float(e["a"]),
                     alpha=float(e["alpha"]), qmin=float(e["qmin"]), qmax=float(e["qmax"]),
                     masa=float(e.get("masa", 0.0)),
                     centro_masa=np.array(e.get("centro_masa", [0, 0, 0]), float),
                     inercia=np.array(e.get("inercia", np.zeros((3, 3))), float),
                     friccion_viscosa=float(e.get("friccion_viscosa", 0.0)),
                     friccion_seca=float(e.get("friccion_seca", 0.0)),
                     inercia_motor=float(e.get("inercia_motor", 0.0))) for e in eslabones],
            gravedad=np.array(datos.get("gravedad", [0, 0, -9.81]), float),
            herramienta=None if datos.get("herramienta") is None
            else np.array(datos["herramienta"], float))
    except (KeyError, TypeError, ValueError, OverflowError) as err:
        raise ValueError("El archivo contiene datos con formato incorrecto.") from err
    errores = robot.validar()
    if errores:
        raise ValueError("\n".join(errores))
    return robot


def _aplicar(dh_df: pd.DataFrame, masas_df: pd.DataFrame, grav: np.ndarray) -> bool:
    """Reconstruye el robot con lo que quedó en las tablas. Devuelve True si algo cambió."""
    rob: Robot = ss().robot
    nuevos: list[Eslabon] = []
    col_t, col_a = f"θ [{uni()}]", f"α [{uni()}]"
    for i in range(len(dh_df)):
        f = dh_df.iloc[i]
        tipo = "R" if str(f["Tipo"]).strip().lower().startswith("gira") else "P"
        conv = a_interno if tipo == "R" else (lambda v: float(v))
        m = masas_df.iloc[i] if i < len(masas_df) else None
        nuevos.append(Eslabon(
            tipo=tipo,
            theta=a_interno(float(f[col_t] or 0)), d=float(f["d [m]"] or 0),
            a=float(f["a [m]"] or 0), alpha=a_interno(float(f[col_a] or 0)),
            qmin=conv(float(f["q mínima"])), qmax=conv(float(f["q máxima"])),
            masa=float(m["masa [kg]"]) if m is not None else 0.0,
            centro_masa=np.array([float(m["cm x"]), float(m["cm y"]), float(m["cm z"])])
            if m is not None else np.zeros(3),
            inercia=np.diag([float(m["Ixx"]), float(m["Iyy"]), float(m["Izz"])]).astype(float)
            if m is not None else np.zeros((3, 3)),
            friccion_viscosa=float(m["fricción viscosa"]) if m is not None else 0.0,
            friccion_seca=float(m["fricción seca"]) if m is not None else 0.0,
            inercia_motor=float(m["inercia del motor"]) if m is not None else 0.0))
    if not nuevos:
        return False
    candidato = Robot(rob.nombre, nuevos, np.asarray(grav, float), rob.herramienta)
    errores = candidato.validar()
    if errores:
        raise ValueError("\n".join(errores))
    antes = (_df_dh(rob).to_numpy().tolist(), _df_masas(rob).to_numpy().tolist(),
             np.asarray(rob.gravedad).tolist())
    rob.eslabones = nuevos
    rob.gravedad = np.asarray(grav, float)
    q = np.asarray(ss().q, float)
    if len(q) != len(nuevos):
        q = np.resize(q, len(nuevos)) if len(q) else np.zeros(len(nuevos))
        q[len(ss().q):] = 0.0
    lims = rob.limites()
    ss().q = np.clip(q, lims[:, 0], lims[:, 1])
    despues = (_df_dh(rob).to_numpy().tolist(), _df_masas(rob).to_numpy().tolist(),
               np.asarray(rob.gravedad).tolist())
    return antes != despues


def tab_editar() -> None:
    rob: Robot = ss().robot
    encabezado("Editar mi robot",
               "Aquí capturas el robot de tu tarea. Puedes agregar o borrar renglones: "
               "cada renglón es una articulación.")
    st.warning("Esta herramienta es una simulación educativa. Antes de conectar hardware real, "
               "verifica límites, unidades, parada de emergencia y supervisión humana.", icon="⚠️")
    pista("Si no sabes de dónde salen estos cuatro números, vuelve a la pestaña <b>Empezar aquí</b>: "
          "ahí está el dibujo que explica θ, d, a y α.")

    st.markdown("**Tabla de Denavit–Hartenberg**")
    st.caption("En una articulación que gira, θ es solo el desplazamiento fijo que se le suma a la "
               "variable. En una que desliza, ese papel lo hace d.")
    dh_df = st.data_editor(
        _df_dh(rob), num_rows="dynamic", hide_index=True, **A_EDITOR,
        key=f"ed_dh_{ss().version}_{int(ss().grados)}",
        column_config={"Tipo": st.column_config.SelectboxColumn(options=["gira", "desliza"], required=True)})

    st.markdown("**Masas e inercias** (solo hacen falta para la pestaña de motores)")
    st.caption("El centro de masa va medido desde el marco del propio eslabón, y el tensor de inercia "
               "está tomado en el centro de masa.")
    masas_df = st.data_editor(_df_masas(rob), num_rows="dynamic", hide_index=True,
                              key=f"ed_m_{ss().version}", **A_EDITOR)

    st.markdown("**¿Hacia dónde jala la gravedad?**")
    grav = vector_entrada(["gx [m/s²]", "gy [m/s²]", "gz [m/s²]"], rob.gravedad, "grav", 0.5)
    st.caption("Para un brazo de pie que trabaja en un plano vertical, la gravedad va en −y. "
               "Para un robot que se ve en 3D, en −z.")

    if st.button("Aplicar los cambios", type="primary"):
        try:
            _aplicar(dh_df, masas_df, grav)
        except ValueError as err:
            st.error(f"No se pueden aplicar los cambios:\n{err}", icon="⚠️")
        else:
            ss().version += 1
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        datos = _datos_robot(rob)
        st.download_button("Guardar mi robot en un archivo", json.dumps(datos, indent=2),
                           file_name="mi_robot.json", mime="application/json",
                           **A_DESCARGA)
    with c2:
        subido = st.file_uploader("Abrir un robot guardado", type="json")
        if subido is not None:
            contenido = subido.getvalue()
            identificador = hashlib.sha256(contenido).hexdigest()
            if ss().get("archivo_robot_cargado") != identificador:
                try:
                    robot_cargado = _robot_desde_datos(json.loads(contenido.decode("utf-8")))
                    ss().robot = robot_cargado
                    ss().nombre = robot_cargado.nombre
                    ss().archivo_robot_cargado = identificador
                    poner_q(np.zeros(robot_cargado.n))
                    st.success("Robot cargado.", icon="✅")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as err:
                    st.error(f"No se pudo leer el archivo: {err}", icon="⚠️")

    with st.expander("Herramienta montada en el extremo (una pinza, un sensor…)"):
        st.caption("Dónde queda la punta de la herramienta respecto al último marco del robot. "
                   "Se incluye en todos los cálculos.")
        v = vector_entrada(["x [m]", "y [m]", "z [m]", f"roll [{uni()}]",
                            f"pitch [{uni()}]", f"yaw [{uni()}]"], [0.0] * 6, "her", 0.01)
        if np.any(np.abs(v) > 1e-12):
            rob.herramienta = homogenea(R_desde_rpy(*[a_interno(x) for x in v[3:]]), v[:3])
            st.success("Herramienta activa: ya está incluida en la posición de la mano.", icon="✅")
        else:
            rob.herramienta = None


# ================================== MAIN ==================================
def main() -> None:
    st.set_page_config(page_title="Laboratorio de robots", page_icon="🦾",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    if _VERSION < VERSION_MINIMA:
        st.error(f"Tu Streamlit es la versión {st.__version__} y este programa necesita al menos la "
                 f"{'.'.join(map(str, VERSION_MINIMA))}. Ciérralo, escribe en la terminal "
                 f"`pip install --upgrade streamlit` y vuelve a abrirlo.", icon="⚠️")
        st.stop()
    iniciar_estado()
    barra_lateral()
    encabezado_principal()

    pestañas = st.tabs(["🚀  Empezar aquí", "📍  Posición de la mano", "🎯  Llegar a un punto",
                        "⚡  Velocidad y singularidades", "💪  Motores y pares",
                        "🎬  Movimiento suave", "🧭  Rotaciones", "🛠️  Editar mi robot"])
    with pestañas[0]:
        tab_empezar(DIAGRAMA_DH)
    with pestañas[1]:
        tab_directa()
    with pestañas[2]:
        tab_inversa_modular(poner_q, vector_entrada)
    with pestañas[3]:
        tab_jacobiano()
    with pestañas[4]:
        tab_dinamica()
    with pestañas[5]:
        tab_trayectorias()
    with pestañas[6]:
        tab_rotaciones()
    with pestañas[7]:
        tab_editar()


def _dentro_de_streamlit() -> bool:
    """¿Nos esta ejecutando Streamlit, o alguien corrio 'python laboratorio_robots.py'?"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:                                                  # noqa: BLE001
        return False


def revisar() -> None:
    """Comprueba que todo lo necesario este instalado y que las cuentas den bien."""
    import platform
    print("\n  REVISION DEL LABORATORIO DE ROBOTS")
    print("  " + "-" * 52)
    print(f"  Python                {platform.python_version()}  ({sys.executable})")
    problemas = []
    if sys.version_info < (3, 9):
        problemas.append("Python es muy viejo: necesitas la version 3.9 o mas nueva.")
    for paquete, minima in (("streamlit", VERSION_MINIMA), ("numpy", (1, 20)),
                            ("plotly", (5, 0)), ("pandas", (1, 3))):
        try:
            modulo = __import__(paquete)
            version = getattr(modulo, "__version__", "?")
            numeros = tuple(int(x) for x in re.findall(r"\d+", version)[:2])
            estado = "bien" if numeros >= minima else f"MUY VIEJO (pide {'.'.join(map(str, minima))})"
            print(f"  {paquete:21s} {version:10s} {estado}")
            if numeros < minima:
                problemas.append(f"Actualiza {paquete}:  pip install --upgrade {paquete}")
        except ImportError:
            print(f"  {paquete:21s} {'NO INSTALADO':10s}")
            problemas.append(f"Falta {paquete}:  pip install {paquete}")

    try:
        robot, postura = catalogo()["Brazo plano de 2 eslabones"]()
        x, y = robot.posicion([0.0, 0.0])[:2]
        par = robot.pares(postura, np.zeros(2), np.zeros(2))
        bien = abs(x - 1.4) < 1e-9 and abs(y) < 1e-9 and abs(par[0]) > 0
        print(f"  cuentas internas      {'correctas' if bien else 'MAL'}")
        if not bien:
            problemas.append("Las cuentas internas no dan: vuelve a descargar el archivo.")
    except Exception as err:                                           # noqa: BLE001
        print(f"  cuentas internas      ERROR: {err}")
        problemas.append(f"Error al calcular: {err}")

    print("  " + "-" * 52)
    if problemas:
        print("  Hay que arreglar esto antes de abrirlo:\n")
        for p in problemas:
            print(f"    - {p}")
        print()
    else:
        print("  Todo listo. Abrelo con:\n\n      streamlit run " + os.path.basename(__file__) + "\n")


def _lanzar() -> None:
    """Si lo abrieron con python, lo relanzamos nosotros con streamlit."""
    import subprocess
    archivo = os.path.basename(__file__)
    print("\n  Este programa se abre con Streamlit: lo estoy lanzando por ti.\n"
          "  En unos segundos se abrira tu navegador en  http://localhost:8501\n"
          "  Si no se abre solo, copia esa direccion y pegala en el navegador.\n\n"
          f"  La proxima vez puedes escribir:   streamlit run {archivo}\n"
          "  Para cerrarlo, regresa aqui y presiona Ctrl+C.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
    except KeyboardInterrupt:
        print("\n  Listo. Hasta la proxima.\n")
    except Exception as err:                                           # noqa: BLE001
        print(f"\n  No se pudo lanzar Streamlit ({err}).\n"
              f"  Revisa que este instalado con:  python {archivo} --revisar\n")


if __name__ == "__main__":
    if _dentro_de_streamlit():
        main()
    elif len(sys.argv) > 1 and sys.argv[1].lstrip("-") in ("revisar", "check", "diagnostico"):
        revisar()
    else:
        _lanzar()
