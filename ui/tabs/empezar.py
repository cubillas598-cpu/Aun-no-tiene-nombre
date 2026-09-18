"""Pestana de introduccion al laboratorio."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import pista, tabla


def tab_empezar(diagrama_dh: str) -> None:
    st.markdown("### Bienvenido")
    st.markdown(
        '<p class="mini">Esto es una calculadora para el curso de cinemática y dinámica de robots. '
        'Sirve para resolver tareas y, sobre todo, para ver qué significan las cuentas.</p>',
        unsafe_allow_html=True)
    pasos = [
        ("1", "Elige un robot", "En la barra de la izquierda hay seis robots típicos ya cargados. "
                                "Si tu tarea trae otro, ve a la pestaña <b>Editar mi robot</b> y escribe su tabla."),
        ("2", "Muévelo", "Los deslizadores cambian cada articulación. El dibujo en 3D y todos los "
                         "resultados se actualizan al instante."),
        ("3", "Abre la pestaña con tu pregunta", "No hay que apretar “calcular”: todo se recalcula solo."),
    ]
    for num, titulo, texto in pasos:
        st.markdown(f'<div class="paso"><div class="num">{num}</div>'
                    f'<div class="txt"><b>{titulo}</b><span class="mini">{texto}</span></div></div>',
                    unsafe_allow_html=True)

    st.divider()
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### ¿Qué pestaña necesito?")
        guia = pd.DataFrame({
            "Si tu pregunta es…": [
                "¿Dónde queda la mano con estos ángulos?",
                "¿Qué ángulos necesito para llegar a este punto?",
                "¿Qué tan rápido se mueve la mano? ¿Estoy en una singularidad?",
                "¿De cuántos N·m tienen que ser mis motores?",
                "¿Cómo hago que se mueva suave de A a B?",
                "Solo quiero convertir entre ángulos y matrices",
                "Necesito capturar el robot de mi tarea",
            ],
            "Ve a": ["Posición de la mano", "Llegar a un punto", "Velocidad y singularidades",
                     "Motores y pares", "Movimiento suave", "Rotaciones", "Editar mi robot"]})
        tabla(guia)
    with der:
        st.markdown("#### Los cuatro números de cada eslabón")
        st.markdown(diagrama_dh, unsafe_allow_html=True)

    st.markdown("""
Cada renglón de la tabla de un robot dice cómo llegar del eslabón anterior al siguiente con
cuatro movimientos, siempre en el mismo orden:

- **θ (theta)** — cuánto giras alrededor del eje viejo. Si la articulación **gira**, éste es el número que mueve el motor.
- **d** — cuánto avanzas a lo largo del eje viejo. Si la articulación **desliza**, éste es el que mueve el actuador.
- **a** — qué tan largo es el eslabón. Siempre es constante.
- **α (alfa)** — cuánto se inclina el eje siguiente respecto al anterior. Casi siempre vale 0 o ±90°: dice si el siguiente eje queda paralelo o perpendicular.
""")
    pista("<b>Truco:</b> si eres nuevo, empieza con el <b>brazo plano de 2 eslabones</b>. "
          "Tiene solo dos motores, se ve completo en el dibujo y sus fórmulas se pueden verificar a mano.")
