"""Conversiones y operaciones puras sobre rotaciones y transformaciones."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

GRADOS = 180.0 / math.pi


def deg(rad: float) -> float:
    """Radianes a grados."""
    return float(rad) * GRADOS


def rad(grados: float) -> float:
    """Grados a radianes."""
    return float(grados) / GRADOS


def rot_x(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def homogenea(R: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Arma una matriz 4x4 a partir de una rotacion y una traslacion."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, float)
    return T


def inversa_homogenea(T: np.ndarray) -> np.ndarray:
    """Invierte una transformacion homogenea usando R transpuesta."""
    R, p = T[:3, :3], T[:3, 3]
    return homogenea(R.T, -R.T @ p)


def rpy_desde_R(R: np.ndarray) -> tuple[float, float, float]:
    """Angulos fijos ZYX: devuelve roll, pitch, yaw."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy < 1e-9:
        return (math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0)
    return (math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy),
            math.atan2(R[1, 0], R[0, 0]))


def R_desde_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def zyz_desde_R(R: np.ndarray) -> tuple[float, float, float]:
    """Angulos de Euler moviles ZYZ."""
    st = math.hypot(R[0, 2], R[1, 2])
    if st < 1e-9:
        return (0.0, 0.0 if R[2, 2] > 0 else math.pi, math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(R[1, 2], R[0, 2]), math.atan2(st, R[2, 2]),
            math.atan2(R[2, 1], -R[2, 0]))


def R_desde_zyz(phi: float, theta: float, psi: float) -> np.ndarray:
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


def eje_angulo_desde_R(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convierte una rotacion a eje y angulo."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if math.pi - ang < 1e-6:
        eje = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        if R[0, 1] < 0:
            eje[1] = -eje[1]
        if R[0, 2] < 0:
            eje[2] = -eje[2]
        return eje, ang
    s = 2.0 * math.sin(ang)
    eje = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return eje, ang


def R_desde_eje_angulo(eje: Sequence[float], ang: float) -> np.ndarray:
    v = np.asarray(eje, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = v / n
    c, s = math.cos(ang), math.sin(ang)
    u = 1.0 - c
    return np.array([
        [x * x * u + c, x * y * u - z * s, x * z * u + y * s],
        [x * y * u + z * s, y * y * u + c, y * z * u - x * s],
        [x * z * u - y * s, y * z * u + x * s, z * z * u + c]], float)


def cuaternion_desde_R(R: np.ndarray) -> np.ndarray:
    """Devuelve el cuaternion en orden [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4,
                         (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, s / 4])


def R_desde_cuaternion(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) or 1.0)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)
