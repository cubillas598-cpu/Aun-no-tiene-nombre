"""Catalogo de robots educativos preconfigurados."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .robot import Eslabon, Robot
from .rotations import rad


def _e(tipo="R", theta=0.0, d=0.0, a=0.0, alpha=0.0, qmin=-180.0, qmax=180.0, grados=True,
       masa=0.0, cm=(0, 0, 0), I=(0, 0, 0), Fv=0.0, Fc=0.0, Im=0.0) -> Eslabon:
    """Atajo para escribir el catalogo con angulos en grados."""
    c = rad if grados else (lambda v: v)
    lim = (c(qmin), c(qmax)) if tipo == "R" else (qmin, qmax)
    return Eslabon(tipo=tipo, theta=c(theta), d=d, a=a, alpha=c(alpha),
                   qmin=lim[0], qmax=lim[1], masa=masa,
                   centro_masa=np.array(cm, float), inercia=np.diag(I).astype(float),
                   friccion_viscosa=Fv, friccion_seca=Fc, inercia_motor=Im)


def catalogo() -> dict[str, Callable[[], tuple[Robot, np.ndarray]]]:
    """Robots listos para usar. Cada entrada devuelve robot y postura inicial."""

    def brazo_2r():
        r = Robot("Brazo plano de 2 eslabones", [
            _e(a=0.8, masa=5, cm=(-0.4, 0, 0), I=(0, 0, 0.12)),
            _e(a=0.6, qmin=-150, qmax=150, masa=3, cm=(-0.3, 0, 0), I=(0, 0, 0.05))],
            gravedad=np.array([0.0, -9.81, 0.0]))
        return r, np.array([rad(35), rad(50)])

    def brazo_3r():
        r = Robot("Brazo plano de 3 eslabones", [
            _e(a=0.7, masa=4, cm=(-0.35, 0, 0), I=(0, 0, 0.09)),
            _e(a=0.5, qmin=-150, qmax=150, masa=2.5, cm=(-0.25, 0, 0), I=(0, 0, 0.04)),
            _e(a=0.3, qmin=-150, qmax=150, masa=1.0, cm=(-0.15, 0, 0), I=(0, 0, 0.01))],
            gravedad=np.array([0.0, -9.81, 0.0]))
        return r, np.array([rad(30), rad(40), rad(-30)])

    def antropomorfico():
        r = Robot("Brazo antropomorfico de 3 ejes", [
            _e(d=0.5, alpha=90, masa=8, cm=(0, 0, -0.2), I=(0.06, 0.06, 0.04)),
            _e(a=0.6, qmin=-115, qmax=115, masa=6, cm=(-0.3, 0, 0), I=(0.01, 0.2, 0.2)),
            _e(a=0.5, qmin=-150, qmax=150, masa=3, cm=(-0.25, 0, 0), I=(0.008, 0.07, 0.07))])
        return r, np.array([rad(25), rad(35), rad(-50)])

    def scara():
        r = Robot("SCARA de 4 ejes", [
            _e(d=0.4, a=0.325, qmin=-130, qmax=130, masa=10, cm=(-0.15, 0, 0), I=(0.05, 0.2, 0.2)),
            _e(a=0.225, alpha=180, qmin=-150, qmax=150, masa=6, cm=(-0.1, 0, 0), I=(0.03, 0.1, 0.1)),
            _e(tipo="P", qmin=0.0, qmax=0.25, masa=2, cm=(0, 0, -0.05), I=(0.01, 0.01, 0.002)),
            _e(masa=0.8, cm=(0, 0, -0.02), I=(0.002, 0.002, 0.001))])
        return r, np.array([rad(30), rad(-45), 0.10, 0.0])

    def cartesiano():
        r = Robot("Robot cartesiano (portico)", [
            _e(tipo="P", alpha=-90, qmin=0.0, qmax=0.8, masa=20, cm=(0, 0, -0.2), I=(0.5, 0.5, 0.2)),
            _e(tipo="P", theta=-90, alpha=-90, qmin=0.0, qmax=0.8, masa=12, cm=(0, 0, -0.15), I=(0.3, 0.3, 0.1)),
            _e(tipo="P", qmin=0.0, qmax=0.8, masa=5, cm=(0, 0, -0.1), I=(0.1, 0.1, 0.05))])
        return r, np.array([0.4, 0.3, 0.5])

    def puma():
        r = Robot("PUMA 560 industrial de 6 ejes", [
            _e(alpha=90, qmin=-160, qmax=160, masa=0.0, I=(0, 0.35, 0), Fv=1.48e-3, Fc=0.395, Im=0.784),
            _e(a=0.4318, qmin=-225, qmax=45, masa=17.4, cm=(-0.3638, 0.006, 0.2275),
               I=(0.13, 0.524, 0.539), Fv=0.817e-3, Fc=0.126, Im=2.30),
            _e(d=0.15005, a=0.0203, alpha=-90, qmin=-45, qmax=225, masa=4.8,
               cm=(-0.0203, -0.0141, 0.07), I=(0.066, 0.086, 0.0125), Fv=1.38e-3, Fc=0.132, Im=0.586),
            _e(d=0.4318, alpha=90, qmin=-110, qmax=170, masa=0.82, cm=(0, 0.019, 0),
               I=(1.8e-3, 1.3e-3, 1.8e-3), Fv=71.2e-6, Fc=11.2e-3, Im=0.191),
            _e(alpha=-90, qmin=-100, qmax=100, masa=0.34, I=(0.3e-3, 0.4e-3, 0.3e-3),
               Fv=82.6e-6, Fc=9.26e-3, Im=0.171),
            _e(qmin=-266, qmax=266, masa=0.09, cm=(0, 0, 0.032), I=(0.15e-3, 0.15e-3, 0.04e-3),
               Fv=36.7e-6, Fc=3.96e-3, Im=0.195)])
        return r, np.array([0.0, rad(-45), rad(60), 0.0, rad(30), 0.0])

    return {"Brazo plano de 2 eslabones": brazo_2r,
            "Brazo plano de 3 eslabones": brazo_3r,
            "Brazo antropomorfico (3 ejes)": antropomorfico,
            "SCARA (4 ejes)": scara,
            "Cartesiano / portico (3 ejes)": cartesiano,
            "PUMA 560 (6 ejes)": puma}
