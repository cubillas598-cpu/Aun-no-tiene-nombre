# Laboratorio de robotica

Aplicacion educativa para estudiar cinematica y dinamica de robots mediante una interfaz interactiva.

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

## Alcance y seguridad

El programa es una simulacion educativa. Los resultados dependen de que la tabla DH, las masas,
los limites y las unidades esten bien configurados. No sustituye la validacion de seguridad,
la parada de emergencia ni la supervision humana al trabajar con un robot real.