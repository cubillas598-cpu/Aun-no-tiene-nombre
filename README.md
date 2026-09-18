# Laboratorio de robotica

Aplicacion educativa para estudiar cinematica y dinamica de robots mediante una interfaz interactiva.

El nucleo de rotaciones y el modelo del robot estan separados en `domain/rotations.py` y
`domain/robot.py` para poder probarlos sin depender de la interfaz Streamlit. La aplicacion
principal sigue siendo `laboratorio_robots.py` mientras se completa la migracion gradual del resto
del dominio.

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