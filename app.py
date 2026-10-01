"""
Interfaz web (Flask) para predecir con los tres modelos de regresión exportados.

Ejecutar:
    pip install -r requirements.txt
    python entrenar_modelos.py     # (solo si no existen los .joblib en modelos/)
    python app.py
Abrir en el navegador: http://127.0.0.1:5000
"""
import json
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.path.join(BASE, "modelos")
GRA_DIR = os.path.join(BASE, "graficas")
RES_DIR = os.path.join(BASE, "resultados")

with open(os.path.join(RES_DIR, "resultados.json"), encoding="utf-8") as f:
    RESULTADOS = json.load(f)

# Etiquetas legibles y restricciones de entrada para cada variable
CAMPOS = {
    "Dia": {"etiqueta": "Día (número de día)", "paso": "1", "min": 1},
    "Inflacion": {"etiqueta": "Inflación diaria (tasa, ej. 0.02 = 2%)", "paso": "any"},
    "Tasa_interes": {"etiqueta": "Tasa de interés diaria (%)", "paso": "any"},
    "Edad": {"etiqueta": "Edad (años)", "paso": "1", "min": 0, "max": 120},
    "IMC": {"etiqueta": "Índice de masa corporal (kg/m²)", "paso": "any", "min": 0},
    "Actividad_Fisica": {"etiqueta": "Actividad física (horas/semana)", "paso": "any", "min": 0},
    "Temperatura": {"etiqueta": "Temperatura (°C)", "paso": "any"},
    "Hora": {"etiqueta": "Hora del día (1 a 24)", "paso": "1", "min": 1, "max": 24},
    "Dia_Semana": {"etiqueta": "Día de la semana (1 = Lunes … 7 = Domingo)", "paso": "1", "min": 1, "max": 7},
}

NOMBRES = {"dolar": "Dólar", "glucosa": "Glucosa", "energia": "Energía"}

# Carga de los modelos exportados (una sola vez al iniciar)
MODELOS = {}
for clave in NOMBRES:
    ruta = os.path.join(MOD_DIR, f"modelo_{clave}.joblib")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}. Ejecute primero: python entrenar_modelos.py")
    MODELOS[clave] = joblib.load(ruta)

app = Flask(__name__)


def dashboard_data():
    """Prepara los datos resumidos para el dashboard principal."""
    modelos = []
    for clave, info in RESULTADOS.items():
        modelos.append({
            "clave": clave,
            "titulo": info["titulo"],
            "n_registros": info["n_registros"],
            "r2": info["metricas"]["prueba"]["R2"],
            "rmse": info["metricas"]["prueba"]["RMSE"],
            "impacto": info["variable_mayor_impacto"],
            "ecuacion": info["ecuacion"],
            "imagen_real": f"/graficas/{clave}_real_vs_predicho.png",
            "imagen_importancia": f"/graficas/{clave}_importancia.png",
            "imagen_correlacion": f"/graficas/{clave}_correlacion.png",
        })
    mejor = max(modelos, key=lambda m: m["r2"], default=None)
    total_registros = sum(m["n_registros"] for m in modelos)
    promedio_r2 = sum(m["r2"] for m in modelos) / len(modelos) if modelos else 0
    return {
        "modelos": modelos,
        "mejor": mejor,
        "total_registros": total_registros,
        "promedio_r2": promedio_r2,
    }


@app.route("/graficas/<path:nombre>")
def servir_graficas(nombre):
    return send_from_directory(GRA_DIR, nombre)


def predecir(clave, valores):
    """Valida los datos de entrada y devuelve (predicción, avisos)."""
    paq = MODELOS[clave]
    fila, avisos = {}, []
    for f in paq["features"]:
        bruto = str(valores.get(f, "")).strip().replace(",", ".")
        if bruto == "":
            raise ValueError(f"Falta el valor de {f}.")
        try:
            v = float(bruto)
        except ValueError:
            raise ValueError(f"El valor de {f} debe ser numérico.")
        reglas = CAMPOS.get(f, {})
        if "min" in reglas and v < reglas["min"]:
            raise ValueError(f"{f} debe ser mayor o igual a {reglas['min']}.")
        if "max" in reglas and v > reglas["max"]:
            raise ValueError(f"{f} debe ser menor o igual a {reglas['max']}.")
        r = paq["rangos"][f]
        if v < r["min"] or v > r["max"]:
            avisos.append(f"{f} = {v:g} está fuera del rango de entrenamiento "
                          f"({r['min']:.4g} – {r['max']:.4g}); la predicción es una extrapolación.")
        fila[f] = v
    X = pd.DataFrame([fila], columns=paq["features"])
    return float(paq["modelo"].predict(X)[0]), avisos


@app.route("/", methods=["GET", "POST"])
def index():
    clave = request.values.get("ejercicio", "dolar")
    if clave not in MODELOS:
        clave = "dolar"
    paq = MODELOS[clave]
    resultado, error, avisos = None, None, []
    valores = {f: request.form.get(f, "") for f in paq["features"]}

    if request.method == "POST" and request.form.get("accion") == "predecir":
        try:
            resultado, avisos = predecir(clave, request.form)
        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html", clave=clave, nombres=NOMBRES, paq=paq, campos=CAMPOS,
        valores=valores, resultado=resultado, error=error, avisos=avisos,
        dashboard=dashboard_data())


@app.route("/api/predecir/<clave>", methods=["POST"])
def api_predecir(clave):
    """Uso: POST /api/predecir/glucosa  {"Edad": 45, "IMC": 27, "Actividad_Fisica": 3}"""
    if clave not in MODELOS:
        return jsonify({"error": "Ejercicio no válido"}), 404
    try:
        pred, avisos = predecir(clave, request.get_json(force=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    paq = MODELOS[clave]
    return jsonify({"ejercicio": clave, "variable": paq["target"],
                    "prediccion": pred, "unidad": paq["unidad"], "avisos": avisos})


if __name__ == "__main__":
    app.run(debug=True)
