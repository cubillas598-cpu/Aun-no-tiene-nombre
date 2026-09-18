# Laboratorio de robotica

Aplicacion educativa para estudiar cinematica y dinamica de robots mediante una interfaz interactiva.

El nucleo matematico esta separado en `domain/rotations.py`, `domain/robot.py`,
`domain/catalog.py`, `domain/inverse.py`, `domain/trajectories.py` y `domain/inertia.py`.
Estos modulos se pueden probar sin depender de la interfaz Streamlit. La aplicacion principal
usa tambien `ui/state.py` para el estado de sesion y `ui/figures.py` para las graficas.
Las pestañas restantes siguen en `laboratorio_robots.py` mientras termina la migracion gradual
de la interfaz.

## Instalacion

```powershell
pip install -r requirements.txt
```

## Ejecucion

```powershell
streamlit run laboratorio_robots.py
```

Tambien puedes revisar las dependencias y las cuentas internas con:

```powershell
python laboratorio_robots.py --revisar
```

Para ejecutar las pruebas preventivas:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Alcance y seguridad

El programa es una simulacion educativa. Los resultados dependen de que la tabla DH, las masas,
los limites y las unidades esten bien configurados. No sustituye la validacion de seguridad,
la parada de emergencia ni la supervision humana al trabajar con un robot real.