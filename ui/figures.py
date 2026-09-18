"""Figuras Plotly de la interfaz del laboratorio."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain.robot import Robot

TINTA = "#171A1F"
SUAVE = "#6B7580"
MARINO = "#1B3A6B"
AMBAR = "#E0862D"
COLOR_X, COLOR_Y, COLOR_Z = "#E0544A", "#2F9E63", "#2C6BD6"
PALETA = ["#1B3A6B", "#E0862D", "#177A4C", "#8E44AD", "#C0392B", "#0E7C86", "#B7950B", "#5D6D7E"]


def figura_robot(robot: Robot, q: Sequence[float], nube: np.ndarray | None = None,
                 altura: int = 620) -> go.Figure:
    Ts = robot.cadena(q)
    puntos = np.array([T[:3, 3] for T in Ts])
    alcance = max(0.3, float(np.max(np.abs(puntos))) * 1.05,
                  max((abs(e.a) + abs(e.d)) for e in robot.eslabones))
    fig = go.Figure()

    if nube is not None and len(nube):
        fig.add_trace(go.Scatter3d(x=nube[:, 0], y=nube[:, 1], z=nube[:, 2], mode="markers",
                                   name="espacio de trabajo", hoverinfo="skip",
                                   marker=dict(size=2, color=AMBAR, opacity=0.16)))

    g = alcance
    paso = max(g / 5.0, 0.05)
    v = -g
    lx, ly, lz = [], [], []
    while v <= g + 1e-9:
        lx += [v, v, None, -g, g, None]
        ly += [-g, g, None, v, v, None]
        lz += [0, 0, None, 0, 0, None]
        v += paso
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", hoverinfo="skip",
                               marker=dict(size=1.7, color=MARINO, opacity=0.12),
                               name="hasta donde alcanza"))

    bx, by, bz = [], [], []
    for i, e in enumerate(robot.eslabones):
        o = Ts[i][:3, 3]
        z = Ts[i][:3, 2]
        dv = e.d + (0.0 if e.es_rotacion else float(q[i]))
        medio = o + z * dv
        fin = Ts[i + 1][:3, 3]
        bx += [o[0], medio[0], fin[0], None]
        by += [o[1], medio[1], fin[1], None]
        bz += [o[2], medio[2], fin[2], None]
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", hoverinfo="skip",
                               line=dict(color="#2A3138", width=9), showlegend=False))

    for tipo, color, etiqueta in (("R", MARINO, "articulación que gira"), ("P", AMBAR, "articulación que desliza")):
        jx, jy, jz = [], [], []
        for i, e in enumerate(robot.eslabones):
            if e.tipo != tipo:
                continue
            o, z = Ts[i][:3, 3], Ts[i][:3, 2]
            h = alcance * 0.075
            jx += [o[0] - z[0] * h, o[0] + z[0] * h, None]
            jy += [o[1] - z[1] * h, o[1] + z[1] * h, None]
            jz += [o[2] - z[2] * h, o[2] + z[2] * h, None]
        if jx:
            fig.add_trace(go.Scatter3d(x=jx, y=jy, z=jz, mode="lines", name=etiqueta,
                                       line=dict(color=color, width=16), hoverinfo="skip"))

    for k, (color, nombre) in enumerate(((COLOR_X, "eje x"), (COLOR_Y, "eje y"), (COLOR_Z, "eje z"))):
        ex, ey, ez = [], [], []
        for j, T in enumerate(Ts):
            largo = alcance * (0.2 if j in (0, len(Ts) - 1) else 0.11)
            o, u = T[:3, 3], T[:3, k]
            ex += [o[0], o[0] + u[0] * largo, None]
            ey += [o[1], o[1] + u[1] * largo, None]
            ez += [o[2], o[2] + u[2] * largo, None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", name=nombre,
                                   line=dict(color=color, width=4), hoverinfo="skip"))

    p = puntos[-1]
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name="la mano",
                               marker=dict(size=7, color=TINTA),
                               hovertemplate="mano<br>x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}<extra></extra>"))

    ejes = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=True,
                title="", tickfont=dict(size=9, color=SUAVE), color=SUAVE)
    fig.update_layout(
        height=altura, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=10, color=SUAVE),
                    bgcolor="rgba(0,0,0,0)"),
        scene=dict(xaxis=ejes, yaxis=ejes, zaxis=ejes, aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)), dragmode="orbit"),
    )
    return fig


def figura_curvas(datos: list[tuple[str, np.ndarray, np.ndarray]], robot: Robot) -> go.Figure:
    fig = make_subplots(rows=len(datos), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[d[0] for d in datos])
    for fila, (_, x, Y) in enumerate(datos, start=1):
        for i in range(robot.n):
            fig.add_trace(go.Scatter(x=x, y=Y[:, i], mode="lines", name=f"eje {i + 1}",
                                     legendgroup=f"eje{i}", showlegend=(fila == 1),
                                     line=dict(color=PALETA[i % len(PALETA)], width=2.2)),
                          row=fila, col=1)
    fig.update_xaxes(title_text="tiempo [s]", row=len(datos), col=1)
    fig.update_layout(height=190 * len(datos), margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(family="Inter", size=11, color=TINTA),
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor="#EDEFEB", zerolinecolor="#DDE0DA")
    fig.update_yaxes(gridcolor="#EDEFEB", zerolinecolor="#DDE0DA")
    for a in fig.layout.annotations:
        a.font.size = 12
        a.font.family = "Space Grotesk"
    return fig
