"""Modelo de eslabones y robot, independiente de la interfaz."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .rotations import eje_angulo_desde_R


@dataclass
class Eslabon:
    """Un eslabon con parametros DH, limites y datos de masa."""
    tipo: str = "R"
    theta: float = 0.0
    d: float = 0.0
    a: float = 0.0
    alpha: float = 0.0
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
        th = self.theta + q if self.es_rotacion else self.theta
        dd = self.d if self.es_rotacion else self.d + q
        return dh(th, dd, self.a, self.alpha)


def dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matriz DH estandar A = Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
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
    herramienta: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.eslabones)

    def firma(self) -> tuple:
        return (tuple((e.tipo, e.theta, e.d, e.a, e.alpha, e.masa, *np.asarray(e.centro_masa).ravel(),
                       *np.asarray(e.inercia).ravel(), e.friccion_viscosa, e.friccion_seca,
                       e.inercia_motor) for e in self.eslabones),
                tuple(np.asarray(self.gravedad).ravel()),
                None if self.herramienta is None else tuple(np.asarray(self.herramienta).ravel()))

    def limites(self) -> np.ndarray:
        return np.array([[e.qmin, e.qmax] for e in self.eslabones], float)

    def validar(self) -> list[str]:
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

    def cadena(self, q: Sequence[float]) -> list[np.ndarray]:
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
        return self.cadena(q)[-1]

    def posicion(self, q: Sequence[float]) -> np.ndarray:
        return self.pose(q)[:3, 3]

    def jacobiano(self, q: Sequence[float]) -> np.ndarray:
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
        J = self.jacobiano(q)
        return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def valores_singulares(self, q: Sequence[float]) -> np.ndarray:
        return np.linalg.svd(self.jacobiano(q), compute_uv=False)

    def rango(self, q: Sequence[float]) -> int:
        return int(np.linalg.matrix_rank(self.jacobiano(q), tol=1e-8))

    def inversa(self, objetivo: np.ndarray, q0: Sequence[float] | None = None,
                solo_posicion: bool = False, lam: float = 0.05,
                max_iter: int = 400, tol_pos: float = 1e-6, tol_ori: float = 1e-5,
                respetar_limites: bool = False) -> dict:
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

    def pares(self, q, qd, qdd, carga: Sequence[float] | None = None,
              con_gravedad: bool = True, con_friccion: bool = True,
              traza: dict | None = None) -> np.ndarray:
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
        w = np.zeros(3)
        wd = np.zeros(3)
        vd = -np.asarray(self.gravedad, float) if con_gravedad else np.zeros(3)
        W, WD, VD, VDC = [], [], [], []
        for i, e in enumerate(self.eslabones):
            Rt = R[i].T
            r = Rt @ pr[i]
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
        n = self.n
        M = np.zeros((n, n))
        cero = np.zeros(n)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            M[:, j] = self.pares(q, cero, ej, con_gravedad=False, con_friccion=False)
        return M

    def vector_coriolis(self, q, qd) -> np.ndarray:
        return self.pares(q, qd, np.zeros(self.n), con_gravedad=False, con_friccion=False)

    def vector_gravedad(self, q) -> np.ndarray:
        cero = np.zeros(self.n)
        return self.pares(q, cero, cero, con_gravedad=True, con_friccion=False)

    def matriz_coriolis(self, q, qd) -> np.ndarray:
        n = self.n
        q = np.asarray(q, float)
        if not np.any(np.asarray(qd, float)):
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
        rng = np.random.default_rng(semilla)
        lims = self.limites()
        Q = rng.uniform(lims[:, 0], lims[:, 1], size=(muestras, self.n))
        return np.array([self.posicion(qi) for qi in Q])
