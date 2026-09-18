"""Componentes visuales reutilizables de la interfaz."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def dice(texto: str) -> None:
    st.markdown(f'<div class="dice">{texto}</div>', unsafe_allow_html=True)


def pista(texto: str, explicar: bool | None = None) -> None:
    if explicar if explicar is not None else st.session_state.get("explicar", True):
        st.markdown(f'<div class="pista">{texto}</div>', unsafe_allow_html=True)


def encabezado(pregunta: str, respuesta: str = "") -> None:
    st.markdown(f"### {pregunta}")
    if respuesta:
        st.markdown(f'<p class="mini">{respuesta}</p>', unsafe_allow_html=True)


def tabla(df: pd.DataFrame, argumentos: dict | None = None) -> None:
    st.dataframe(df, hide_index=True, **(argumentos or {}))
