# Regresión lineal múltiple – CRISP-DM (Dólar, Glucosa, Energía)

## Estructura
```
proyecto_regresion/
├── data/                    # dolar_data.csv, glucosa_data.csv, energia_data.csv
├── entrenar_modelos.py      # Fases CRISP-DM: comprensión, preparación, modelado, evaluación, exportación
├── app.py                   # Interfaz web Flask (selección de ejercicio + predicción)
├── templates/index.html     # Página de la interfaz
├── modelos/                 # modelo_dolar.joblib, modelo_glucosa.joblib, modelo_energia.joblib
├── graficas/                # dispersión, correlación, real vs. predicho, importancia
├── resultados/              # resultados.json y resumen.md (métricas e interpretación)
├── resultados_consola.txt   # salida completa del entrenamiento
└── requirements.txt
```

## Cómo ejecutar
```bash
pip install -r requirements.txt
python entrenar_modelos.py   # entrena, evalúa, grafica y exporta los 3 modelos
python app.py                # abre http://127.0.0.1:5000
```

## Desplegar la página estática en Vercel
La página `index.html` de la raíz funciona de forma independiente: ejecuta las
predicciones en el navegador con los coeficientes de `resultados/resultados.json`.
No necesita Flask, Python ni una función de servidor en Vercel.

1. Sube el proyecto a un repositorio de GitHub.
2. En Vercel, importa el repositorio, selecciona `proyecto_regresion` como
	**Root Directory** y usa el preset **Other**.
3. Deja vacíos el comando de compilación y el directorio de salida; despliega la
	raíz del proyecto, donde están `index.html`, `resultados/` y `graficas/`.

Para probarla localmente, ejecuta `python -m http.server 8000` desde la carpeta
que contiene `index.html` y abre `http://localhost:8000`. No abras el HTML
directamente como archivo, porque el navegador debe cargar `resultados/resultados.json` mediante HTTP.

## Guardar y cargar un modelo (joblib)
```python
import joblib, pandas as pd
paquete = joblib.load("modelos/modelo_glucosa.joblib")
X = pd.DataFrame([{"Edad": 45, "IMC": 27, "Actividad_Fisica": 3}])
print(paquete["modelo"].predict(X))      # -> nivel de glucosa estimado
```
Cada `.joblib` guarda un diccionario con el modelo entrenado, el orden de las
variables, la unidad, las métricas, los coeficientes y los rangos de los datos.
Con `pickle` sería equivalente: `pickle.dump(obj, open("m.pkl","wb"))` y
`pickle.load(open("m.pkl","rb"))`; joblib es más eficiente con arreglos de NumPy.

## API (opcional)
`POST /api/predecir/<dolar|glucosa|energia>` con JSON, por ejemplo:
`{"Temperatura": 30, "Hora": 18, "Dia_Semana": 3}`
