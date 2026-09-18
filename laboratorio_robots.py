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
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# ==============================================================================
#  PARTE 1 - MATEMATICAS
#  Las operaciones puras viven en domain/ para poder probarlas sin Streamlit.
# ==============================================================================

from domain.rotations import (GRADOS, R_desde_cuaternion, R_desde_eje_angulo,
                               R_desde_rpy, R_desde_zyz, cuaternion_desde_R, deg,
                               eje_angulo_desde_R, homogenea, inversa_homogenea,
                               rad, rot_x, rot_y, rot_z, rpy_desde_R, zyz_desde_R)


# --------------------------------------------------------------- eslabon y robot
@dataclass
class Eslabon:
    """Un eslabon: sus cuatro parametros DH, sus limites y sus datos de masa."""
    tipo: str = "R"            # "R" = rotacion, "P" = prismatica
    theta: float = 0.0         # rad. En una junta R es el desplazamiento que se suma a q
    d: float = 0.0             # m.   En una junta P es el desplazamiento que se suma a q
    a: float = 0.0             # m.   longitud del eslabon
    alpha: float = 0.0         # rad. torsion entre ejes consecutivos
    qmin: float = -math.pi
    qmax: float = math.pi
    masa: float = 0.0
    centro_masa: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inercia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    friccion_viscosa: float = 0.0
    friccion_seca: float = 0.0
    inercia_motor: float = 0.0

    @property
    def es_rotacion(self) -> bool:
        return self.tipo == "R"

    def matriz(self, q: float) -> np.ndarray:
        """Matriz A del eslabon para un valor de la variable articular."""
        th = self.theta + q if self.es_rotacion else self.theta
        dd = self.d if self.es_rotacion else self.d + q
        return dh(th, dd, self.a, self.alpha)


def dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """A = Rot_z(theta) . Tras_z(d) . Tras_x(a) . Rot_x(alpha)"""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]], float)


@dataclass
class Robot:
    nombre: str
    eslabones: list[Eslabon]
    gravedad: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    herramienta: np.ndarray | None = None      # 4x4 del ultimo marco a la punta

    # -------------------------------------------------------------- basicos
    @property
    def n(self) -> int:
        return len(self.eslabones)

    def firma(self) -> tuple:
        """Resumen inmutable del robot: sirve para no repetir calculos ya hechos."""
        return (tuple((e.tipo, e.theta, e.d, e.a, e.alpha, e.masa, *np.asarray(e.centro_masa).ravel(),
                       *np.asarray(e.inercia).ravel(), e.friccion_viscosa, e.friccion_seca,
                       e.inercia_motor) for e in self.eslabones),
                tuple(np.asarray(self.gravedad).ravel()),
                None if self.herramienta is None else tuple(np.asarray(self.herramienta).ravel()))

    def limites(self) -> np.ndarray:
        return np.array([[e.qmin, e.qmax] for e in self.eslabones], float)

    def validar(self) -> list[str]:
        """Devuelve errores de configuracion antes de usar el robot en calculos."""
        errores = []
        gravedad = np.asarray(self.gravedad, float)
        if gravedad.shape != (3,) or not np.all(np.isfinite(gravedad)):
            errores.append("La gravedad debe tener tres componentes finitas.")
        for i, eslabon in enumerate(self.eslabones, start=1):
            prefijo = f"Eslabon {i}: "
            if eslabon.tipo not in ("R", "P"):
                errores.append(prefijo + "el tipo debe ser R o P.")
            if not np.isfinite([eslabon.qmin, eslabon.qmax]).all() or eslabon.qmin >= eslabon.qmax:
                errores.append(prefijo + "q minima debe ser menor que q maxima.")
            valores = [eslabon.theta, eslabon.d, eslabon.a, eslabon.alpha, eslabon.masa,
                       eslabon.friccion_viscosa, eslabon.friccion_seca, eslabon.inercia_motor]
            if not np.isfinite(valores).all():
                errores.append(prefijo + "contiene parametros no finitos.")
            if eslabon.masa < 0:
                errores.append(prefijo + "la masa no puede ser negativa.")
            centro = np.asarray(eslabon.centro_masa, float)
            inercia = np.asarray(eslabon.inercia, float)
            if centro.shape != (3,) or not np.all(np.isfinite(centro)):
                errores.append(prefijo + "el centro de masa debe tener tres valores finitos.")
            if inercia.shape != (3, 3) or not np.all(np.isfinite(inercia)):
                errores.append(prefijo + "la inercia debe ser una matriz 3x3 finita.")
            elif not np.allclose(inercia, inercia.T, atol=1e-9):
                errores.append(prefijo + "la inercia debe ser simetrica.")
            elif np.min(np.linalg.eigvalsh(inercia)) < -1e-9:
                errores.append(prefijo + "la inercia debe ser semidefinida positiva.")
        if self.herramienta is not None:
            herramienta = np.asarray(self.herramienta, float)
            if herramienta.shape != (4, 4) or not np.all(np.isfinite(herramienta)):
                errores.append("La herramienta debe ser una matriz 4x4 finita.")
            elif not np.allclose(herramienta[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                errores.append("La ultima fila de la herramienta debe ser [0, 0, 0, 1].")
        return errores

    def q_inicial(self) -> np.ndarray:
        return np.zeros(self.n)

    # --------------------------------------------------- cinematica directa
    def cadena(self, q: Sequence[float]) -> list[np.ndarray]:
        """[T0_0=I, T0_1, ..., T0_n] con la herramienta ya incluida al final."""
        q = np.asarray(q, float)
        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Se esperaban {self.n} variables articulares finitas.")
        Ts = [np.eye(4)]
        for e, qi in zip(self.eslabones, q):
            Ts.append(Ts[-1] @ e.matriz(float(qi)))
        if self.herramienta is not None:
            Ts[-1] = Ts[-1] @ self.herramienta
        return Ts

    def pose(self, q: Sequence[float]) -> np.ndarray:
        """Matriz 4x4 del extremo."""
        return self.cadena(q)[-1]

    def posicion(self, q: Sequence[float]) -> np.ndarray:
        return self.pose(q)[:3, 3]

    # ------------------------------------------------------------ jacobiano
    def jacobiano(self, q: Sequence[float]) -> np.ndarray:
        """Jacobiano geometrico 6xn en el marco base: [v; omega] = J qpunto."""
        Ts = self.cadena(q)
        pn = Ts[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i, e in enumerate(self.eslabones):
            T = Ts[i]
            z = T[:3, 2]
            p = T[:3, 3]
            if e.es_rotacion:
                J[:3, i] = np.cross(z, pn - p)
                J[3:, i] = z
            else:
                J[:3, i] = z
        return J

    def manipulabilidad(self, q: Sequence[float]) -> float:
        """Vale cero exactamente en las singularidades."""
        J = self.jacobiano(q)
        return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def valores_singulares(self, q: Sequence[float]) -> np.ndarray:
        return np.linalg.svd(self.jacobiano(q), compute_uv=False)

    def rango(self, q: Sequence[float]) -> int:
        return int(np.linalg.matrix_rank(self.jacobiano(q), tol=1e-8))

    # ----------------------------------------------------- cinematica inversa
    def inversa(self, objetivo: np.ndarray, q0: Sequence[float] | None = None,
                solo_posicion: bool = False, lam: float = 0.05,
                max_iter: int = 400, tol_pos: float = 1e-6, tol_ori: float = 1e-5,
                respetar_limites: bool = False) -> dict:
        """
        Minimos cuadrados amortiguados:  dq = J^T (J J^T + lam^2 I)^-1 e
        Sirve para cualquier robot y encuentra la solucion mas cercana a q0.
        """
        q = np.array(self.q_inicial() if q0 is None else q0, float)
        pd, Rd = objetivo[:3, 3], objetivo[:3, :3]
        lims = self.limites()
        err_p = err_o = float("inf")
        it = 0
        for it in range(max_iter + 1):
            T = self.pose(q)
            e = pd - T[:3, 3]
            err_p = float(np.linalg.norm(e))
            if solo_posicion:
                err_o = 0.0
                error, J = e, self.jacobiano(q)[:3]
            else:
                eje, ang = eje_angulo_desde_R(Rd @ T[:3, :3].T)
                err_o = abs(ang)
                error, J = np.concatenate([e, eje * ang]), self.jacobiano(q)
            if err_p < tol_pos and err_o < tol_ori:
                break
            A = J @ J.T + (lam ** 2) * np.eye(J.shape[0])
            try:
                dq = J.T @ np.linalg.solve(A, error)
            except np.linalg.LinAlgError:
                break
            q = q + dq
            if respetar_limites:
                q = np.clip(q, lims[:, 0], lims[:, 1])
        return {"q": q, "iteraciones": it, "error_pos": err_p, "error_ori": err_o,
                "exito": err_p < tol_pos and err_o < tol_ori}

    # ------------------------------------------------------------- dinamica
    def pares(self, q, qd, qdd, carga: Sequence[float] | None = None,
              con_gravedad: bool = True, con_friccion: bool = True,
              traza: dict | None = None) -> np.ndarray:
        """
        Newton-Euler recursivo. Devuelve el par (o fuerza) de cada actuador.
        'carga' es la fuerza y el momento que la mano ejerce sobre el entorno,
        en coordenadas del marco base: [Fx, Fy, Fz, Mx, My, Mz].
        """
        n = self.n
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        qdd = np.asarray(qdd, float)
        if any(v.shape != (n,) or not np.all(np.isfinite(v)) for v in (q, qd, qdd)):
            raise ValueError(f"q, qd y qdd deben tener {n} valores finitos.")
        if carga is not None:
            carga = np.asarray(carga, float)
            if carga.shape != (6,) or not np.all(np.isfinite(carga)):
                raise ValueError("La carga debe tener seis valores finitos.")
        z0 = np.array([0.0, 0.0, 1.0])

        A = [e.matriz(float(q[i])) for i, e in enumerate(self.eslabones)]
        R = [a[:3, :3] for a in A]
        pr = [a[:3, 3] for a in A]

        # --- hacia adelante: de la base a la mano, propagando el movimiento
        w = np.zeros(3)
        wd = np.zeros(3)
        vd = -np.asarray(self.gravedad, float) if con_gravedad else np.zeros(3)
        W, WD, VD, VDC = [], [], [], []
        for i, e in enumerate(self.eslabones):
            Rt = R[i].T
            r = Rt @ pr[i]                       # vector origen i-1 -> origen i, en el marco i
            if e.es_rotacion:
                wi = Rt @ (w + qd[i] * z0)
                wdi = Rt @ (wd + qdd[i] * z0 + qd[i] * np.cross(w, z0))
                vdi = Rt @ vd + np.cross(wdi, r) + np.cross(wi, np.cross(wi, r))
            else:
                wi = Rt @ w
                wdi = Rt @ wd
                vdi = (Rt @ (vd + qdd[i] * z0) + 2 * qd[i] * np.cross(wi, Rt @ z0)
                       + np.cross(wdi, r) + np.cross(wi, np.cross(wi, r)))
            rc = np.asarray(e.centro_masa, float)
            vdci = vdi + np.cross(wdi, rc) + np.cross(wi, np.cross(wi, rc))
            W.append(wi); WD.append(wdi); VD.append(vdi); VDC.append(vdci)
            w, wd, vd = wi, wdi, vdi

        # --- hacia atras: de la mano a la base, propagando fuerzas
        h = np.zeros(6) if carga is None else np.asarray(carga, float)
        Rn = self.cadena(q)[-1][:3, :3]
        f_sig = Rn.T @ h[:3]
        m_sig = Rn.T @ h[3:]
        R_sig = np.eye(3)
        tau = np.zeros(n)
        F, M = [None] * n, [None] * n
        for i in range(n - 1, -1, -1):
            e = self.eslabones[i]
            Rt = R[i].T
            r = Rt @ pr[i]
            rc = np.asarray(e.centro_masa, float)
            Rf = R_sig @ f_sig
            fi = Rf + e.masa * VDC[i]
            I = np.asarray(e.inercia, float)
            mi = (-np.cross(fi, r + rc) + R_sig @ m_sig + np.cross(Rf, rc)
                  + I @ WD[i] + np.cross(W[i], I @ W[i]))
            zi = Rt @ z0
            t = float(mi @ zi) if e.es_rotacion else float(fi @ zi)
            t += e.inercia_motor * qdd[i]
            if con_friccion:
                t += e.friccion_viscosa * qd[i] + e.friccion_seca * math.copysign(1.0, qd[i]) * (qd[i] != 0)
            tau[i] = t
            F[i], M[i] = fi, mi
            f_sig, m_sig, R_sig = fi, mi, R[i]

        if traza is not None:
            traza.update(omega=W, omega_punto=WD, a_origen=VD, a_centro_masa=VDC,
                         fuerza=F, momento=M, tau=tau)
        return tau

    def matriz_inercia(self, q) -> np.ndarray:
        """M(q). Cada columna se obtiene pidiendo aceleracion unitaria en un solo eje."""
        n = self.n
        M = np.zeros((n, n))
        cero = np.zeros(n)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            M[:, j] = self.pares(q, cero, ej, con_gravedad=False, con_friccion=False)
        return M

    def vector_coriolis(self, q, qd) -> np.ndarray:
        """C(q,qpunto) qpunto: efectos centrifugos y de Coriolis."""
        return self.pares(q, qd, np.zeros(self.n), con_gravedad=False, con_friccion=False)

    def vector_gravedad(self, q) -> np.ndarray:
        """g(q): lo que cuesta solo sostener el brazo, sin moverlo."""
        cero = np.zeros(self.n)
        return self.pares(q, cero, cero, con_gravedad=True, con_friccion=False)

    def matriz_coriolis(self, q, qd) -> np.ndarray:
        """C(q,qpunto) completa, por simbolos de Christoffel (derivadas numericas de M)."""
        n = self.n
        q = np.asarray(q, float)
        if not np.any(np.asarray(qd, float)):     # quieto: C vale cero y no hay que calcular nada
            return np.zeros((n, n))
        h = 1e-6
        M0 = self.matriz_inercia(q)
        dM = []
        for k in range(n):
            qk = q.copy()
            qk[k] += h
            dM.append((self.matriz_inercia(qk) - M0) / h)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = sum(0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qd[k] for k in range(n))
        return C

    def energia(self, q, qd, M: np.ndarray | None = None) -> dict:
        M = self.matriz_inercia(q) if M is None else M
        cinetica = 0.5 * float(np.asarray(qd) @ M @ np.asarray(qd))
        Ts = self.cadena(q)
        potencial = 0.0
        for i, e in enumerate(self.eslabones):
            p = Ts[i + 1][:3, 3] + Ts[i + 1][:3, :3] @ np.asarray(e.centro_masa, float)
            potencial -= e.masa * float(np.asarray(self.gravedad) @ p)
        return {"cinetica": cinetica, "potencial": potencial, "total": cinetica + potencial, "M": M}

    def espacio_trabajo(self, muestras: int = 3000, semilla: int = 0) -> np.ndarray:
        """Nube de puntos alcanzables, sorteando posturas al azar dentro de los limites."""
        rng = np.random.default_rng(semilla)
        lims = self.limites()
        Q = rng.uniform(lims[:, 0], lims[:, 1], size=(muestras, self.n))
        return np.array([self.posicion(qi) for qi in Q])


# ---------------------------------------------- cinematica inversa con formulas
def inversa_2R(a1: float, a2: float, x: float, y: float) -> list[dict]:
    """Brazo plano de dos eslabones. Da las dos soluciones: codo arriba y codo abajo."""
    r2 = x * x + y * y
    c2 = (r2 - a1 * a1 - a2 * a2) / (2 * a1 * a2) if a1 * a2 else 2.0
    if abs(c2) > 1 + 1e-9:
        return []
    c2 = max(-1.0, min(1.0, c2))
    s2 = math.sqrt(max(1 - c2 * c2, 0.0))
    sols = []
    for signo, nombre in ((1, "arriba"), (-1, "abajo")):
        th2 = math.atan2(signo * s2, c2)
        th1 = math.atan2(y, x) - math.atan2(a2 * math.sin(th2), a1 + a2 * math.cos(th2))
        sols.append({"q": np.array([th1, th2]), "codo": nombre})
    return sols


def inversa_3R_plano(a1, a2, a3, x, y, phi) -> list[dict]:
    """Brazo plano de tres eslabones cuando ademas se pide la orientacion phi."""
    xm = x - a3 * math.cos(phi)
    ym = y - a3 * math.sin(phi)
    out = []
    for s in inversa_2R(a1, a2, xm, ym):
        th1, th2 = s["q"]
        out.append({"q": np.array([th1, th2, phi - th1 - th2]), "codo": s["codo"]})
    return out


def inversa_antropomorfico(a1, a2, a3, d1, x, y, z) -> list[dict]:
    """Tres ejes: giro de base + hombro + codo. Hasta cuatro soluciones."""
    out = []
    for atras in (False, True):
        th1 = math.atan2(y, x) + (math.pi if atras else 0.0)
        r = (-math.hypot(x, y) if atras else math.hypot(x, y)) - a1
        s = z - d1
        for sol in inversa_2R(a2, a3, r, s):
            out.append({"q": np.array([th1, sol["q"][0], sol["q"][1]]),
                        "codo": sol["codo"], "hombro": "atras" if atras else "adelante"})
    return out


# -------------------------------------------------------------- trayectorias
def cubica(q0, qf, v0, vf, tf) -> Callable[[float], np.ndarray]:
    """Velocidad continua. Es el perfil mas sencillo que sirve de verdad."""
    a0, a1 = q0, v0
    a2 = (3 * (qf - q0) - (2 * v0 + vf) * tf) / tf ** 2
    a3 = (2 * (q0 - qf) + (v0 + vf) * tf) / tf ** 3

    def f(t):
        return np.array([a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3,
                         a1 + 2 * a2 * t + 3 * a3 * t ** 2,
                         2 * a2 + 6 * a3 * t])
    return f


def quintica(q0, qf, v0, vf, ac0, acf, tf) -> Callable[[float], np.ndarray]:
    """Arranca y termina con aceleracion cero: lo mas suave para la mecanica."""
    d, T = qf - q0, tf
    a0, a1, a2 = q0, v0, ac0 / 2
    a3 = (20 * d - (8 * vf + 12 * v0) * T - (3 * ac0 - acf) * T ** 2) / (2 * T ** 3)
    a4 = (-30 * d + (14 * vf + 16 * v0) * T + (3 * ac0 - 2 * acf) * T ** 2) / (2 * T ** 4)
    a5 = (12 * d - 6 * (vf + v0) * T + (acf - ac0) * T ** 2) / (2 * T ** 5)

    def f(t):
        return np.array([a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5,
                         a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4,
                         2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3])
    return f


def trapezoidal(q0, qf, tf, aceleracion) -> Callable[[float], np.ndarray]:
    """Acelera, viaja a velocidad constante y frena. Es el perfil industrial."""
    d = qf - q0
    D = abs(d)
    signo = math.copysign(1.0, d) if d else 1.0
    if D < 1e-12:
        f = lambda t: np.array([q0, 0.0, 0.0])                     # noqa: E731
        f.tc, f.v_crucero, f.aceleracion, f.insuficiente = 0.0, 0.0, 0.0, False
        return f
    minima = 4 * D / tf ** 2
    insuficiente = aceleracion < minima
    a = max(aceleracion, minima * 1.0000001)
    tc = tf / 2 - math.sqrt(max(a * a * tf * tf - 4 * a * D, 0.0)) / (2 * a)
    vc = a * tc

    def f(t):
        t = min(max(t, 0.0), tf)
        if t <= tc:
            return np.array([q0 + signo * 0.5 * a * t * t, signo * a * t, signo * a])
        if t <= tf - tc:
            return np.array([q0 + signo * vc * (t - tc / 2), signo * vc, 0.0])
        u = tf - t
        return np.array([qf - signo * 0.5 * a * u * u, signo * a * u, -signo * a])

    f.tc, f.v_crucero, f.aceleracion, f.insuficiente = tc, signo * vc, signo * a, insuficiente
    return f


# ------------------------------------------------- tensores de inercia utiles
def inercia_varilla(m, L):
    """Varilla delgada con su eje largo sobre x."""
    return np.diag([0.0, m * L ** 2 / 12, m * L ** 2 / 12])


def inercia_cilindro(m, r, L):
    """Cilindro con su eje sobre x."""
    return np.diag([m * r ** 2 / 2, m * (3 * r ** 2 + L ** 2) / 12, m * (3 * r ** 2 + L ** 2) / 12])


def inercia_prisma(m, a, b, c):
    return np.diag([m * (b ** 2 + c ** 2) / 12, m * (a ** 2 + c ** 2) / 12, m * (a ** 2 + b ** 2) / 12])


def inercia_esfera(m, r):
    return np.diag([2 * m * r ** 2 / 5] * 3)


FORMAS = {
    "Varilla delgada (eje en x)": (inercia_varilla, ["largo L [m]"]),
    "Cilindro (eje en x)": (inercia_cilindro, ["radio r [m]", "largo L [m]"]),
    "Prisma rectangular": (inercia_prisma, ["lado a [m]", "lado b [m]", "lado c [m]"]),
    "Esfera maciza": (inercia_esfera, ["radio r [m]"]),
}


def steiner(I: np.ndarray, m: float, d: Sequence[float]) -> np.ndarray:
    """Teorema de los ejes paralelos: traslada el tensor a otro punto."""
    d = np.asarray(d, float)
    return I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))


# ------------------------------------------------------- catalogo de robots
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
    """Robots listos para usar. Cada entrada devuelve (robot, postura inicial)."""

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


# ==============================================================================
#  PARTE 2 - INTERFAZ
#  Streamlit para la pantalla, Plotly para el robot en 3D.
# ==============================================================================

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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


# --------------------------------------------------------------- utilidades UI
def ss():
    return st.session_state


def _cargar(nombre: str) -> None:
    robot, q0 = catalogo()[nombre]()
    ss().robot = robot
    ss().q = np.asarray(q0, float)
    ss().nombre = nombre
    ss()._elegido = nombre
    ss().version = ss().get("version", 0) + 1
    ss().nube = None


def iniciar_estado() -> None:
    if "robot" not in ss():
        ss().unidad_ang = "grados"
        ss().ver_explicaciones = True
        ss().grados = True
        ss().explicar = True
        _cargar("Brazo antropomorfico (3 ejes)")
    # los controles guardan su valor con una clave propia: asi no se atoran al cambiarlos
    ss().grados = ss().get("unidad_ang", "grados") == "grados"
    ss().explicar = bool(ss().get("ver_explicaciones", True))


def uni() -> str:
    return "°" if ss().grados else "rad"


def a_pantalla(valor: float) -> float:
    return deg(valor) if ss().grados else valor


def a_interno(valor: float) -> float:
    return rad(valor) if ss().grados else valor


def unidad_junta(e: Eslabon) -> str:
    return uni() if e.es_rotacion else "m"


def unidad_par(e: Eslabon) -> str:
    return "N·m" if e.es_rotacion else "N"


def n_fmt(x: float, dec: int = 3) -> str:
    x = float(x)
    if abs(x) < 5e-7:
        x = 0.0
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{dec}f}"


def latex_matriz(M: np.ndarray, nombre: str = "", dec: int = 3) -> str:
    cuerpo = r" \\ ".join(" & ".join(n_fmt(v, dec) for v in fila) for fila in np.atleast_2d(M))
    izq = f"{nombre} = " if nombre else ""
    return izq + r"\begin{bmatrix}" + cuerpo + r"\end{bmatrix}"


def dice(texto: str) -> None:
    st.markdown(f'<div class="dice">{texto}</div>', unsafe_allow_html=True)


def pista(texto: str) -> None:
    if ss().explicar:
        st.markdown(f'<div class="pista">{texto}</div>', unsafe_allow_html=True)


def encabezado(pregunta: str, respuesta: str = "") -> None:
    st.markdown(f"### {pregunta}")
    if respuesta:
        st.markdown(f'<p class="mini">{respuesta}</p>', unsafe_allow_html=True)


def tabla(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True, **A_TABLA)


# ------------------------------------------------------------- robot en 3D
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

    # eslabones, siguiendo la construccion DH: primero d sobre z viejo, luego a sobre x nuevo
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

    # marcadores de articulacion sobre su eje de giro o de deslizamiento
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

    # marcos de referencia de cada eslabon
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


# ============================== PESTAÑA: EMPEZAR ==============================
def tab_empezar() -> None:
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
        st.markdown(DIAGRAMA_DH, unsafe_allow_html=True)

    st.markdown("""
Cada renglón de la tabla de un robot dice cómo llegar del eslabón anterior al siguiente con
cuatro movimientos, siempre en el mismo orden:

- **θ (theta)** — cuánto giras alrededor del eje viejo. Si la articulación **gira**, éste es el número que mueve el motor.
- **d** — cuánto avanzas a lo largo del eje viejo. Si la articulación **desliza**, éste es el que mueve el actuador.
- **a** — qué tan largo es el eslabón. Siempre es constante.
- **α (alfa)** — cuánto se inclina el eje siguiente respecto al anterior. Casi siempre vale 0 o ±90°: dice si el
  siguiente eje queda paralelo o perpendicular.
""")
    pista("<b>Truco:</b> si eres nuevo, empieza con el <b>brazo plano de 2 eslabones</b>. "
          "Tiene solo dos motores, se ve completo en el dibujo y sus fórmulas se pueden verificar a mano.")


# ======================== PESTAÑA: CINEMATICA DIRECTA ========================
def tab_directa() -> None:
    rob: Robot = ss().robot
    q = ss().q
    Ts = rob.cadena(q)
    T = Ts[-1]
    p = T[:3, 3]
    encabezado("¿Dónde está la mano?",
               "Le das los ángulos de cada motor y te dice en qué punto del espacio queda el extremo.")
    dice(f"Con la postura de ahora, la mano está en <b>x = {p[0]:.3f} m</b>, "
         f"<b>y = {p[1]:.3f} m</b>, <b>z = {p[2]:.3f} m</b>. "
         f"Todo eso está guardado en una sola matriz de 4×4.")
    pista("Se multiplica una matriz por eslabón: <b>T = A₁·A₂·…·Aₙ</b>. En el resultado, la última columna "
          "es la <b>posición</b> de la mano y el bloque de 3×3 de la izquierda es su <b>orientación</b>, "
          "o sea hacia dónde apunta.")

    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("**La matriz que lo resume todo**")
        st.latex(latex_matriz(T, "T", 3))
    with der:
        st.markdown("**La misma orientación, escrita de todas las formas**")
        roll, pitch, yaw = rpy_desde_R(T[:3, :3])
        phi, tht, psi = zyz_desde_R(T[:3, :3])
        eje, ang = eje_angulo_desde_R(T[:3, :3])
        cu = cuaternion_desde_R(T[:3, :3])
        tabla(pd.DataFrame({
            "Forma": ["Posición [m]", f"RPY fijos ZYX [{uni()}]", f"Euler ZYZ [{uni()}]",
                      f"Eje y ángulo [{uni()}]", "Cuaternión"],
            "Valores": [
                f"x {n_fmt(p[0])}   y {n_fmt(p[1])}   z {n_fmt(p[2])}",
                f"{n_fmt(a_pantalla(roll), 2)}   {n_fmt(a_pantalla(pitch), 2)}   {n_fmt(a_pantalla(yaw), 2)}",
                f"{n_fmt(a_pantalla(phi), 2)}   {n_fmt(a_pantalla(tht), 2)}   {n_fmt(a_pantalla(psi), 2)}",
                f"eje ({', '.join(n_fmt(v, 2) for v in eje)})   ángulo {n_fmt(a_pantalla(ang), 2)}",
                "   ".join(n_fmt(v) for v in cu)]}))

    with st.expander("Ver el procedimiento eslabón por eslabón (para copiar a la tarea)"):
        for i, e in enumerate(rob.eslabones):
            th = e.theta + q[i] if e.es_rotacion else e.theta
            dd = e.d if e.es_rotacion else e.d + q[i]
            extra = " (el desplazamiento fijo más q)" if (e.es_rotacion and e.theta) else (" = q" if e.es_rotacion else "")
            st.markdown(f"**Eslabón {i + 1}** · θ = {n_fmt(a_pantalla(th), 2)}{uni()}{extra} · "
                        f"d = {n_fmt(dd)} m · a = {n_fmt(e.a)} m · α = {n_fmt(a_pantalla(e.alpha), 2)}{uni()}")
            c1, c2 = st.columns(2)
            with c1:
                st.latex(latex_matriz(e.matriz(float(q[i])), f"A_{{{i + 1}}}", 3))
            with c2:
                st.latex(latex_matriz(Ts[i + 1], f"T^0_{{{i + 1}}}", 3))
        st.code("\n".join(" ".join(f"{v:10.5f}" for v in fila) for fila in T),
                language=None)


# ======================== PESTAÑA: CINEMATICA INVERSA ========================
def tab_inversa() -> None:
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
        tab_empezar()
    with pestañas[1]:
        tab_directa()
    with pestañas[2]:
        tab_inversa()
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
