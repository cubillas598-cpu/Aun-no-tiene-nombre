"""Pestana de cinematica inversa."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from domain.inverse import inversa_2R, inversa_3R_plano, inversa_antropomorfico
from domain.robot import Robot
from domain.rotations import R_desde_rpy, homogenea, rpy_desde_R
from ui.components import dice, encabezado, pista, tabla
from ui.state import a_interno, a_pantalla, n_fmt, ss, uni, unidad_junta


def tab_inversa(poner_q, vector_entrada) -> None:
    rob: Robot = ss().robot
    q = ss().q
    encabezado("¿Qué ángulos necesito para llegar a un punto?",
               "Al revés que la anterior: tú dices a dónde quieres llegar y el programa busca los ángulos.")
    pista("El método <b>numérico</b> arranca de la postura actual y va corrigiendo con el jacobiano hasta "
          "acercarse; funciona con cualquier robot y da <b>una</b> solución, la más cercana. Las "
          "<b>fórmulas cerradas</b> de más abajo dan <b>todas</b> las soluciones de un jalón, pero solo "
          "existen para algunas geometrías.")

    if st.button("Copiar la posición donde está ahora", help="Rellena las casillas con la pose actual."):
        T = rob.pose(q)
        r, pt, y = rpy_desde_R(T[:3, :3])
        for nombre, valor in zip(["obj_x [m]", "obj_y [m]", "obj_z [m]"], T[:3, 3]):
            ss()[f"obj_{nombre}"] = float(valor)
        for nombre, valor in zip([f"roll [{uni()}]", f"pitch [{uni()}]", f"yaw [{uni()}]"],
                                 [a_pantalla(r), a_pantalla(pt), a_pantalla(y)]):
            ss()[f"ori_{nombre}"] = float(valor)
        st.rerun()

    T_act = rob.pose(q)
    r0, p0, y0 = rpy_desde_R(T_act[:3, :3])
    st.markdown("**¿A qué punto quieres que llegue la mano?**")
    destino = vector_entrada(["obj_x [m]", "obj_y [m]", "obj_z [m]"], T_act[:3, 3], "obj", 0.05)
    modo = st.radio("¿Qué debe cumplir?",
                    ["Solo llegar al punto", "Llegar al punto y además apuntar hacia un lado"],
                    index=0 if rob.n < 6 else 1, horizontal=False, key=f"ik_modo_{rob.n}")
    solo_pos = modo.startswith("Solo")
    if solo_pos:
        orient = np.array([r0, p0, y0])
    else:
        vals = vector_entrada([f"roll [{uni()}]", f"pitch [{uni()}]", f"yaw [{uni()}]"],
                              [a_pantalla(r0), a_pantalla(p0), a_pantalla(y0)], "ori", 5.0)
        orient = np.array([a_interno(v) for v in vals])
    if rob.n < 6 and not solo_pos:
        st.info(f"Este robot tiene {rob.n} articulaciones. Con menos de 6 no se puede imponer al mismo "
                f"tiempo el punto y la orientación, así que el resultado será el mejor compromiso posible.",
                icon="ℹ️")

    with st.expander("Ajustes del buscador (no suelen hacer falta)"):
        c1, c2, c3, c4 = st.columns(4)
        lam = c1.number_input("Amortiguamiento λ", value=0.05, step=0.01, format="%.3f")
        iters = c2.number_input("Intentos máximos", value=400, step=50, min_value=10)
        tol = c3.number_input("Precisión [m]", value=1e-6, step=1e-6, format="%.1e")
        limitar = c4.checkbox("Respetar los topes", value=True)

    sol = rob.inversa(homogenea(R_desde_rpy(*orient), destino), q0=q, solo_posicion=solo_pos,
                      lam=float(lam), max_iter=int(iters), tol_pos=float(tol),
                      respetar_limites=bool(limitar))
    qs = sol["q"]

    if sol["exito"]:
        cuantos = ("Ya estaba en la solución." if sol["iteraciones"] == 0
                   else f"Resuelto en {sol['iteraciones']} "
                   f"{'intento' if sol['iteraciones'] == 1 else 'intentos'}.")
        st.success(f"{cuantos} La mano queda a {sol['error_pos'] * 1000:.3f} mm del punto pedido.", icon="✅")
    else:
        motivo = ("Es normal: con este robot no se pueden imponer las seis coordenadas a la vez. "
                  "Cambia la opción a *solo llegar al punto* y llegará exacto."
                  if (not solo_pos and rob.n < 6) else
                  "Lo más probable es que el punto esté fuera de su alcance, o que los topes de las "
                  "articulaciones no lo dejen. Prueba a mover un poco el robot y vuelve a intentar.")
        st.warning(f"Se quedó a {sol['error_pos'] * 1000:.1f} mm del objetivo. {motivo}", icon="⚠️")

    df = pd.DataFrame({
        "Eje": [f"{i + 1}" for i in range(rob.n)],
        "Qué hace": ["gira" if e.es_rotacion else "desliza" for e in rob.eslabones],
        "Valor que necesita": [n_fmt(a_pantalla(v) if e.es_rotacion else v, 3)
                               for v, e in zip(qs, rob.eslabones)],
        "Unidad": [unidad_junta(e) for e in rob.eslabones],
        "¿Dentro de sus topes?": ["sí" if e.qmin - 1e-9 <= v <= e.qmax + 1e-9 else "NO"
                                  for v, e in zip(qs, rob.eslabones)]})
    tabla(df)
    if st.button("Mover el robot a esta solución", type="primary"):
        poner_q(qs)
        st.rerun()

    st.divider()
    st.markdown("#### Todas las soluciones, con fórmulas exactas")
    pista("Un mismo punto casi siempre se alcanza de varias maneras: con el <b>codo arriba</b> o "
          "<b>abajo</b>, girando la base hacia <b>adelante</b> o hacia <b>atrás</b>. El robot real "
          "escoge la que quede más cerca de donde está y respete sus topes.")
    geo = st.selectbox("Geometría", ["Brazo plano de 2 eslabones",
                                     "Brazo plano de 3 eslabones (con orientación)",
                                     "Antropomórfico de 3 ejes"])
    if geo.startswith("Brazo plano de 2"):
        v = vector_entrada(["a1 [m]", "a2 [m]", "x [m]", "y [m]"], [0.8, 0.6, 1.0, 0.4], "f2r", 0.05)
        sols, nombres = inversa_2R(*v), ["θ1", "θ2"]
    elif geo.startswith("Brazo plano de 3"):
        v = vector_entrada(["a1 [m]", "a2 [m]", "a3 [m]", "x [m]", "y [m]", f"φ [{uni()}]"],
                           [0.7, 0.5, 0.3, 1.0, 0.4, 0.0], "f3r", 0.05)
        sols = inversa_3R_plano(v[0], v[1], v[2], v[3], v[4], a_interno(v[5]))
        nombres = ["θ1", "θ2", "θ3"]
    else:
        v = vector_entrada(["desfase a1 [m]", "brazo a2 [m]", "antebrazo a3 [m]", "altura d1 [m]",
                            "x [m]", "y [m]", "z [m]"], [0.0, 0.6, 0.5, 0.5, 0.6, 0.3, 0.7], "fant", 0.05)
        sols = inversa_antropomorfico(*v)
        nombres = ["θ1", "θ2", "θ3"]

    if not sols:
        st.warning("Ese punto queda fuera del alcance de esas longitudes, así que no existe solución real. "
                   "La distancia al destino tiene que quedar entre |a1 − a2| y a1 + a2.", icon="⚠️")
    else:
        filas = []
        for k, s in enumerate(sols, 1):
            fila = {"Solución": k}
            for nb, val in zip(nombres, s["q"]):
                fila[f"{nb} [{uni()}]"] = round(a_pantalla(val), 3)
            fila["Configuración"] = f"codo {s['codo']}" + (f", hombro {s['hombro']}" if "hombro" in s else "")
            filas.append(fila)
        dice(f"Hay <b>{len(sols)} formas distintas</b> de llegar a ese punto. Todas son correctas.")
        tabla(pd.DataFrame(filas))
        if len(sols) > 1 and abs(sols[0]["q"][1] - sols[1]["q"][1]) < 1e-6:
            st.warning("Las soluciones salieron iguales: el brazo está totalmente estirado o totalmente "
                       "plegado, que es justo una postura singular.", icon="⚠️")
