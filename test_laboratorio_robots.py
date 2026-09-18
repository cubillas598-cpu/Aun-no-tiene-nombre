import numpy as np
import pytest

import laboratorio_robots as app
from domain.robot import Robot as DomainRobot
from domain.rotations import (R_desde_cuaternion, R_desde_rpy, cuaternion_desde_R,
                               homogenea, inversa_homogenea, rad, rpy_desde_R)
from domain.inertia import inercia_cilindro, inercia_esfera, inercia_prisma, inercia_varilla
from domain.trajectories import cubica, quintica, trapezoidal


def test_validacion_rechaza_q_incompleto():
    assert app.Robot is DomainRobot
    robot, q0 = app.catalogo()["Brazo plano de 2 eslabones"]()
    assert robot.validar() == []
    assert robot.posicion(q0).shape == (3,)
    with pytest.raises(ValueError):
        robot.pose([0.0])


def test_json_conserva_herramienta_y_rechaza_limites_invalidos():
    robot, _ = app.catalogo()["Brazo plano de 2 eslabones"]()
    robot.herramienta = app.homogenea(np.eye(3), [0.1, 0.2, 0.3])
    restaurado = app._robot_desde_datos(app._datos_robot(robot))
    np.testing.assert_allclose(restaurado.herramienta, robot.herramienta)

    datos = app._datos_robot(robot)
    datos["eslabones"][0]["qmin"] = 2
    datos["eslabones"][0]["qmax"] = 1
    with pytest.raises(ValueError):
        app._robot_desde_datos(datos)

    with pytest.raises(ValueError):
        app._robot_desde_datos([])


def test_rotaciones_conservan_pose_y_transformacion():
    R = R_desde_rpy(0.2, -0.4, 0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(R_desde_rpy(*rpy_desde_R(R)), R, atol=1e-9)
    assert np.allclose(R_desde_cuaternion(cuaternion_desde_R(R)), R, atol=1e-9)

    T = homogenea(R, [0.3, -0.2, 0.5])
    assert np.allclose(T @ inversa_homogenea(T), np.eye(4), atol=1e-9)
    assert np.isclose(rad(180), np.pi)


def test_trayectorias_terminan_en_el_objetivo():
    assert np.isclose(cubica(0, 1, 0, 0, 2)(2)[0], 1)
    assert np.isclose(quintica(0, 1, 0, 0, 0, 0, 2)(2)[0], 1)
    assert np.isclose(trapezoidal(0, 1, 2, 1)(2)[0], 1)


def test_modelos_de_inercia_producen_matrices_validas():
    for matriz in (inercia_varilla(1, 2), inercia_cilindro(1, 0.1, 2),
                   inercia_prisma(1, 1, 2, 3), inercia_esfera(1, 0.1)):
        assert matriz.shape == (3, 3)
        assert np.allclose(matriz, matriz.T)
        assert np.min(np.linalg.eigvalsh(matriz)) >= -1e-12
