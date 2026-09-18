"""Pestana para editar, validar y guardar robots."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from domain.robot import Eslabon, Robot
from domain.rotations import R_desde_rpy, homogenea
from ui.components import encabezado, pista
from ui.state import a_interno, a_pantalla, n_fmt, ss, uni


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
            herramienta=None if datos.get("herramienta") is None else np.array(datos["herramienta"], float))
    except (KeyError, TypeError, ValueError, OverflowError) as err:
        raise ValueError("El archivo contiene datos con formato incorrecto.") from err
    errores = robot.validar()
    if errores:
        raise ValueError("\n".join(errores))
    return robot


def _aplicar(dh_df: pd.DataFrame, masas_df: pd.DataFrame, grav: np.ndarray) -> bool:
    rob: Robot = ss().robot
    nuevos: list[Eslabon] = []
    col_t, col_a = f"θ [{uni()}]", f"α [{uni()}]"
    for i in range(len(dh_df)):
        fila = dh_df.iloc[i]
        tipo = "R" if str(fila["Tipo"]).strip().lower().startswith("gira") else "P"
        conv = a_interno if tipo == "R" else (lambda v: float(v))
        masa = masas_df.iloc[i] if i < len(masas_df) else None
        nuevos.append(Eslabon(
            tipo=tipo, theta=a_interno(float(fila[col_t] or 0)), d=float(fila["d [m]"] or 0),
            a=float(fila["a [m]"] or 0), alpha=a_interno(float(fila[col_a] or 0)),
            qmin=conv(float(fila["q mínima"])), qmax=conv(float(fila["q máxima"])),
            masa=float(masa["masa [kg]"]) if masa is not None else 0.0,
            centro_masa=np.array([float(masa["cm x"]), float(masa["cm y"]), float(masa["cm z"])])
            if masa is not None else np.zeros(3),
            inercia=np.diag([float(masa["Ixx"]), float(masa["Iyy"]), float(masa["Izz"])]).astype(float)
            if masa is not None else np.zeros((3, 3)),
            friccion_viscosa=float(masa["fricción viscosa"]) if masa is not None else 0.0,
            friccion_seca=float(masa["fricción seca"]) if masa is not None else 0.0,
            inercia_motor=float(masa["inercia del motor"]) if masa is not None else 0.0))
    if not nuevos:
        return False
    candidato = Robot(rob.nombre, nuevos, np.asarray(grav, float), rob.herramienta)
    errores = candidato.validar()
    if errores:
        raise ValueError("\n".join(errores))
    rob.eslabones = nuevos
    rob.gravedad = np.asarray(grav, float)
    q = np.asarray(ss().q, float)
    if len(q) != len(nuevos):
        q = np.resize(q, len(nuevos)) if len(q) else np.zeros(len(nuevos))
        q[len(ss().q):] = 0.0
    lims = rob.limites()
    ss().q = np.clip(q, lims[:, 0], lims[:, 1])
    return True


def tab_editar(vector_entrada, argumentos_editor: dict, argumentos_descarga: dict, poner_q) -> None:
    rob: Robot = ss().robot
    encabezado("Editar mi robot", "Aquí capturas el robot de tu tarea. Puedes agregar o borrar renglones: "
               "cada renglón es una articulación.")
    st.warning("Esta herramienta es una simulación educativa. Antes de conectar hardware real, "
               "verifica límites, unidades, parada de emergencia y supervisión humana.", icon="⚠️")
    pista("Si no sabes de dónde salen estos cuatro números, vuelve a la pestaña <b>Empezar aquí</b>: "
          "ahí está el dibujo que explica θ, d y α.")

    st.markdown("**Tabla de Denavit–Hartenberg**")
    st.caption("En una articulación que gira, θ es el desplazamiento fijo. En una que desliza, ese papel lo hace d.")
    dh_df = st.data_editor(_df_dh(rob), num_rows="dynamic", hide_index=True, **argumentos_editor,
                           key=f"ed_dh_{ss().version}_{int(ss().grados)}",
                           column_config={"Tipo": st.column_config.SelectboxColumn(options=["gira", "desliza"], required=True)})
    st.markdown("**Masas e inercias** (solo hacen falta para la pestaña de motores)")
    masas_df = st.data_editor(_df_masas(rob), num_rows="dynamic", hide_index=True,
                              key=f"ed_m_{ss().version}", **argumentos_editor)
    st.markdown("**¿Hacia dónde jala la gravedad?**")
    grav = vector_entrada(["gx [m/s²]", "gy [m/s²]", "gz [m/s²]"], rob.gravedad, "grav", 0.5)

    if st.button("Aplicar los cambios", type="primary"):
        try:
            _aplicar(dh_df, masas_df, grav)
        except ValueError as err:
            st.error(f"No se pueden aplicar los cambios:\n{err}", icon="⚠️")
        else:
            ss().version += 1
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Guardar mi robot en un archivo", json.dumps(_datos_robot(rob), indent=2),
                           file_name="mi_robot.json", mime="application/json", **argumentos_descarga)
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
        st.caption("Dónde queda la punta de la herramienta respecto al último marco del robot.")
        v = vector_entrada(["x [m]", "y [m]", "z [m]", f"roll [{uni()}]", f"pitch [{uni()}]", f"yaw [{uni()}]"],
                           [0.0] * 6, "her", 0.01)
        rob.herramienta = homogenea(R_desde_rpy(*[a_interno(x) for x in v[3:]]), v[:3]) if np.any(np.abs(v) > 1e-12) else None
        if rob.herramienta is not None:
            st.success("Herramienta activa: ya está incluida en la posición de la mano.", icon="✅")
