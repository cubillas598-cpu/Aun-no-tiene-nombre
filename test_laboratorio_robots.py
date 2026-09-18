import numpy as np
import pytest

import laboratorio_robots as app


def test_validacion_rechaza_q_incompleto():
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
