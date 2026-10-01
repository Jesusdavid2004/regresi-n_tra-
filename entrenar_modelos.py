"""
MINERÍA DE DATOS - Regresión lineal múltiple con metodología CRISP-DM
Ejercicios: 1) Precio del dólar  2) Nivel de glucosa  3) Consumo de energía

Ejecutar:  python entrenar_modelos.py
Entradas:  data/dolar_data.csv, data/glucosa_data.csv, data/energia_data.csv
Salidas:
  modelos/modelo_*.joblib   -> modelos exportados (los usa app.py)
  graficas/*.png            -> dispersión, correlación y real vs. predicho
  resultados/resultados.json, resultados/resumen.md -> métricas e interpretación
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MOD_DIR = os.path.join(BASE, "modelos")
GRA_DIR = os.path.join(BASE, "graficas")
RES_DIR = os.path.join(BASE, "resultados")
for d in (MOD_DIR, GRA_DIR, RES_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# FASE 1 - COMPRENSIÓN DEL NEGOCIO: objetivo de cada ejercicio
# ---------------------------------------------------------------------------
EJERCICIOS = {
    "dolar": {
        "titulo": "Predicción del precio del dólar",
        "archivo": "dolar_data.csv",
        "features": ["Dia", "Inflacion", "Tasa_interes"],
        "target": "Precio_Dolar",
        "unidad": "pesos",
    },
    "glucosa": {
        "titulo": "Predicción del nivel de glucosa en sangre",
        "archivo": "glucosa_data.csv",
        "features": ["Edad", "IMC", "Actividad_Fisica"],
        "target": "Nivel_Glucosa",
        "unidad": "mg/dL",
    },
    "energia": {
        "titulo": "Predicción del consumo de energía eléctrica",
        "archivo": "energia_data.csv",
        "features": ["Temperatura", "Hora", "Dia_Semana"],
        "target": "Consumo_Energia",
        "unidad": "kWh",
    },
}

COLOR = "#2a6f97"
COLOR_LINEA = "#c8553d"


def titulo(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# FASE 2 - COMPRENSIÓN DE LOS DATOS
# ---------------------------------------------------------------------------
def comprender_datos(df, cfg):
    print(f"Registros: {len(df)} | Columnas: {list(df.columns)}")
    print("\nValores nulos por columna:\n", df.isnull().sum().to_string())
    print(f"\nFilas duplicadas: {df.duplicated().sum()}")
    print("\nEstadística descriptiva:\n", df.describe().round(4).to_string())
    corr = df[cfg["features"] + [cfg["target"]]].corr()
    print("\nCorrelación de Pearson con la variable objetivo:\n",
          corr[cfg["target"]].drop(cfg["target"]).round(4).to_string())


# ---------------------------------------------------------------------------
# FASE 3 - PREPARACIÓN DE LOS DATOS
# ---------------------------------------------------------------------------
def preparar_datos(df, cfg):
    cols = cfg["features"] + [cfg["target"]]
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Columnas faltantes en {cfg['archivo']}: {faltan}")
    n0 = len(df)
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    print(f"Limpieza: {n0} -> {len(df)} registros (sin nulos, no numéricos ni duplicados)")
    X, y = df[cfg["features"]], df[cfg["target"]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Partición 80/20 -> entrenamiento: {len(X_tr)}, prueba: {len(X_te)}")
    return df, X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# FASE 4 - MODELADO
# ---------------------------------------------------------------------------
def inferencia(modelo, X, y):
    """Errores estándar, estadístico t y p-valor de cada coeficiente (MCO)."""
    Xm = np.column_stack([np.ones(len(X)), X.values])
    resid = y.values - modelo.predict(X)
    n, k = Xm.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(Xm.T @ Xm)))
    betas = np.r_[modelo.intercept_, modelo.coef_]
    t = betas / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    return se, t, p


def modelar(X_tr, y_tr, cfg):
    modelo = LinearRegression().fit(X_tr, y_tr)
    se, t, p = inferencia(modelo, X_tr, y_tr)
    # Coeficiente estandarizado (beta): cambio en desviaciones estándar de y por
    # una desviación estándar de X -> permite comparar variables con distinta escala.
    beta = modelo.coef_ * X_tr.std().values / y_tr.std()
    coefs = [{
        "variable": f,
        "coeficiente": float(modelo.coef_[i]),
        "error_std": float(se[i + 1]),
        "t": float(t[i + 1]),
        "p_valor": float(p[i + 1]),
        "beta_estandarizado": float(beta[i]),
    } for i, f in enumerate(cfg["features"])]
    print(f"Ecuación: {ecuacion(cfg, modelo.intercept_, coefs)}")
    print(pd.DataFrame(coefs).set_index("variable").round(6).to_string())
    return modelo, coefs


def ecuacion(cfg, b0, coefs):
    return f"{cfg['target']} = {b0:.4f} " + " ".join(
        f"{'+' if c['coeficiente'] >= 0 else '-'} {abs(c['coeficiente']):.4f}·{c['variable']}"
        for c in coefs)


# ---------------------------------------------------------------------------
# FASE 5 - EVALUACIÓN
# ---------------------------------------------------------------------------
def evaluar(modelo, X_tr, y_tr, X_te, y_te):
    out = {}
    for nombre, X, y in (("entrenamiento", X_tr, y_tr), ("prueba", X_te, y_te)):
        pred = modelo.predict(X)
        mse = mean_squared_error(y, pred)
        out[nombre] = {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
                       "MAE": float(mean_absolute_error(y, pred)),
                       "R2": float(r2_score(y, pred))}
    n, k = len(X_tr), X_tr.shape[1]
    r2 = out["entrenamiento"]["R2"]
    out["entrenamiento"]["R2_ajustado"] = float(1 - (1 - r2) * (n - 1) / (n - k - 1))
    print(pd.DataFrame(out).round(4).to_string())
    return out


def interpretar(cfg, coefs, metricas):
    """Interpretación automática a partir de los resultados obtenidos."""
    t, u = cfg["target"], cfg["unidad"]
    lineas = []
    for c in coefs:
        sentido = "aumenta" if c["coeficiente"] > 0 else "disminuye"
        signif = ("significativo (p < 0,05)" if c["p_valor"] < 0.05
                  else "NO significativo (p ≥ 0,05)")
        lineas.append(
            f"- {c['variable']}: por cada unidad adicional, {t} {sentido} "
            f"{abs(c['coeficiente']):.4f} {u} (resto constante). {signif}, "
            f"p = {c['p_valor']:.3g}; beta estandarizado = {c['beta_estandarizado']:.3f}.")
    orden = sorted(coefs, key=lambda c: abs(c["beta_estandarizado"]), reverse=True)
    lineas.append(f"- Mayor impacto: {orden[0]['variable']}. Orden: "
                  + " > ".join(c["variable"] for c in orden) + ".")
    m = metricas["prueba"]
    lineas.append(f"- Prueba: R² = {m['R2']:.4f} ({m['R2']*100:.1f}% de la variabilidad explicada), "
                  f"MSE = {m['MSE']:.4f}, RMSE = {m['RMSE']:.4f} {u}.")
    return orden[0]["variable"], "\n".join(lineas)


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def estilo(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def graficar(df, cfg, clave, modelo, X_te, y_te):
    feats, t = cfg["features"], cfg["target"]
    corr = df[feats + [t]].corr()
    muestra = df.sample(min(len(df), 2500), random_state=SEED)  # legibilidad

    # 1) Dispersión de cada variable independiente vs. la dependiente + recta
    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4.3))
    for ax, f in zip(axes, feats):
        ax.scatter(muestra[f], muestra[t], s=10, alpha=0.45, color=COLOR, edgecolor="none")
        m, b = np.polyfit(df[f], df[t], 1)
        xs = np.linspace(df[f].min(), df[f].max(), 100)
        ax.plot(xs, m * xs + b, color=COLOR_LINEA, lw=2, label=f"tendencia (r = {corr.loc[f, t]:.3f})")
        ax.set_xlabel(f); ax.set_ylabel(f"{t} ({cfg['unidad']})")
        ax.set_title(f"{f} vs {t}", fontsize=11)
        ax.legend(loc="best", fontsize=9, frameon=False)
        estilo(ax)
    fig.suptitle(cfg["titulo"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(GRA_DIR, f"{clave}_dispersion.png"), dpi=130)
    plt.close(fig)

    # 2) Matriz de correlación
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Matriz de correlación – {cfg['titulo']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(GRA_DIR, f"{clave}_correlacion.png"), dpi=130)
    plt.close(fig)

    # 3) Valores reales vs. predichos en el conjunto de prueba
    pred = modelo.predict(X_te)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(y_te, pred, s=12, alpha=0.5, color=COLOR, edgecolor="none")
    lo, hi = min(y_te.min(), pred.min()), max(y_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=COLOR_LINEA, lw=2, ls="--", label="predicción perfecta")
    ax.set_xlabel(f"{t} real"); ax.set_ylabel(f"{t} predicho")
    ax.set_title(f"Real vs. predicho (prueba) – R² = {r2_score(y_te, pred):.3f}", fontsize=11)
    ax.legend(frameon=False); estilo(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRA_DIR, f"{clave}_real_vs_predicho.png"), dpi=130)
    plt.close(fig)

    # 4) Importancia relativa (|beta estandarizado|)
    return corr


def graficar_importancia(cfg, clave, coefs):
    orden = sorted(coefs, key=lambda c: abs(c["beta_estandarizado"]))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    vals = [c["beta_estandarizado"] for c in orden]
    ax.barh([c["variable"] for c in orden], vals,
            color=[COLOR if v >= 0 else COLOR_LINEA for v in vals])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#555", lw=0.8)
    lim = max(abs(v) for v in vals) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Coeficiente estandarizado (beta)")
    ax.set_title(f"Importancia de variables – {clave}", fontsize=11)
    estilo(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRA_DIR, f"{clave}_importancia.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FASE 6 - DESPLIEGUE: exportación del modelo con joblib
# ---------------------------------------------------------------------------
def exportar(clave, cfg, modelo, df, metricas, coefs):
    paquete = {
        "modelo": modelo,                    # objeto LinearRegression entrenado
        "features": cfg["features"],         # orden de las columnas de entrada
        "target": cfg["target"],
        "unidad": cfg["unidad"],
        "titulo": cfg["titulo"],
        "metricas": metricas,
        "coeficientes": coefs,
        "intercepto": float(modelo.intercept_),
        "rangos": {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                       "media": float(df[f].mean())} for f in cfg["features"]},
    }
    ruta = os.path.join(MOD_DIR, f"modelo_{clave}.joblib")
    joblib.dump(paquete, ruta)

    # Verificación: se vuelve a cargar y se predice con valores promedio
    cargado = joblib.load(ruta)
    ejemplo = pd.DataFrame([{f: v["media"] for f, v in cargado["rangos"].items()}])
    print(f"Modelo exportado en: {ruta}")
    print(f"Carga verificada -> predicción con valores promedio: "
          f"{cargado['modelo'].predict(ejemplo)[0]:.4f} {cfg['unidad']}")


def main():
    resultados, md = {}, ["# Resumen de resultados\n"]
    for clave, cfg in EJERCICIOS.items():
        titulo(f"EJERCICIO: {cfg['titulo'].upper()}")
        df = pd.read_csv(os.path.join(DATA_DIR, cfg["archivo"]))

        print("\n--- Fase 2: Comprensión de los datos ---")
        comprender_datos(df, cfg)
        print("\n--- Fase 3: Preparación de los datos ---")
        df, X_tr, X_te, y_tr, y_te = preparar_datos(df, cfg)
        print("\n--- Fase 4: Modelado (regresión lineal múltiple) ---")
        modelo, coefs = modelar(X_tr, y_tr, cfg)
        print("\n--- Fase 5: Evaluación ---")
        metricas = evaluar(modelo, X_tr, y_tr, X_te, y_te)
        mayor, texto = interpretar(cfg, coefs, metricas)
        print("\nInterpretación:\n" + texto)
        corr = graficar(df, cfg, clave, modelo, X_te, y_te)
        graficar_importancia(cfg, clave, coefs)
        print("\n--- Fase 6: Despliegue (exportación del modelo) ---")
        exportar(clave, cfg, modelo, df, metricas, coefs)

        resultados[clave] = {
            "titulo": cfg["titulo"], "n_registros": int(len(df)),
            "ecuacion": ecuacion(cfg, modelo.intercept_, coefs),
            "intercepto": float(modelo.intercept_), "coeficientes": coefs,
            "metricas": metricas, "variable_mayor_impacto": mayor,
            "correlaciones": corr[cfg["target"]].drop(cfg["target"]).round(4).to_dict(),
            "correlaciones_entre_x": corr.loc[cfg["features"], cfg["features"]].round(4).to_dict(),
        }
        md += [f"\n## {cfg['titulo']}\n", f"Registros: {len(df)}\n",
               f"`{resultados[clave]['ecuacion']}`\n", texto, ""]

    with open(os.path.join(RES_DIR, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RES_DIR, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    titulo("Listo: modelos/ (joblib), graficas/ (png), resultados/ (json, md)")


if __name__ == "__main__":
    main()
