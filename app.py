import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

SEED = 42
COLS_CERO_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Predicción de diabetes tipo 2", page_icon="🩺", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_diabetes.joblib"), joblib.load("columnas.joblib")


@st.cache_data
def cargar_datos():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns="Outcome").copy()
    X[COLS_CERO_NAN] = X[COLS_CERO_NAN].replace(0, np.nan)
    y = df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    return df, X_te, y_te


modelo, columnas = cargar_modelo()
df, X_test, y_test = cargar_datos()

# Variables: etiqueta, mínimo, máximo, valor por defecto, paso, ¿puede faltar?
CAMPOS = {
    "Pregnancies": ("Número de embarazos", 0, 20, 2, 1, False),
    "Glucose": ("Glucosa (mg/dL)", 40, 300, 120, 1, True),
    "BloodPressure": ("Presión diastólica (mm Hg)", 20, 130, 70, 1, True),
    "SkinThickness": ("Pliegue cutáneo del tríceps (mm)", 5, 100, 25, 1, True),
    "Insulin": ("Insulina a 2 h (µU/mL)", 10, 900, 80, 1, True),
    "BMI": ("Índice de masa corporal (kg/m²)", 10.0, 70.0, 30.0, 0.1, True),
    "DiabetesPedigreeFunction": ("Función de pedigrí de diabetes", 0.05, 2.5, 0.4, 0.01, False),
    "Age": ("Edad (años)", 21, 100, 33, 1, False),
}

st.title("🩺 Predicción de diabetes tipo 2 a partir de indicadores clínicos")
st.warning("**Prototipo académico.** Esta aplicación no sustituye el diagnóstico ni la "
           "decisión de un profesional de la salud.")

tab1, tab2, tab3, tab4 = st.tabs(["Proyecto", "Exploración (EDA)", "Predicción", "Métricas del modelo"])

# ---------------------------------------------------------------- Proyecto
with tab1:
    st.header("Presentación del proyecto")
    st.markdown("""
**Objetivo:** construir un modelo de clasificación supervisada binaria que, a partir de
indicadores clínicos básicos, prediga si una paciente presenta diabetes (1) o no (0).

**Integrante:** Juan Ortega
**Curso:** Minería de Datos, Universidad de la Costa (CUC)

**Fuente de los datos:** Pima Indians Diabetes Database, publicada por el NIDDK
(Smith et al., 1988). 768 registros de mujeres de al menos 21 años de ascendencia Pima,
con 8 variables predictoras y la variable objetivo `Outcome`.

**Modelo final:** regresión logística dentro de un pipeline (imputación por mediana,
escalamiento estándar y modelo con pesos de clase balanceados).
""")

# --------------------------------------------------------------------- EDA
with tab2:
    st.header("Exploración de datos")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución de Outcome")
        conteo = df["Outcome"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["No diabética (0)", "Diabética (1)"], conteo.values, color=["#4C72B0", "#DD8452"])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 5, f"{v} ({v / len(df):.1%})", ha="center")
        ax.set_ylabel("Pacientes")
        st.pyplot(fig)

    with c2:
        st.subheader("Glucosa según diagnóstico")
        fig, ax = plt.subplots(figsize=(4, 3))
        g = df[df["Glucose"] > 0]
        for k, color, nombre in [(0, "#4C72B0", "No diabética"), (1, "#DD8452", "Diabética")]:
            ax.hist(g[g["Outcome"] == k]["Glucose"], bins=25, alpha=0.6, color=color, label=nombre)
        ax.set_xlabel("Glucosa (mg/dL)")
        ax.set_ylabel("Pacientes")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Tasa de diabetes por categoría de IMC")
    d = df[df["BMI"] > 0].copy()
    d["Categoría IMC"] = pd.cut(d["BMI"], [0, 25, 30, 100], labels=["< 25", "25–30", "> 30"], right=False)
    tasa = d.groupby("Categoría IMC", observed=True)["Outcome"].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(tasa.index.astype(str), tasa.values, color="#DD8452")
    ax.set_ylabel("% con diabetes")
    st.pyplot(fig)
    st.caption("Son asociaciones observadas en los datos, no relaciones causales.")

# --------------------------------------------------------------- Predicción
with tab3:
    st.header("Predicción")
    modo = st.radio("Origen de los datos", ["Ingresar datos manualmente",
                                            "Usar un registro del conjunto de prueba"], horizontal=True)

    if modo == "Ingresar datos manualmente":
        st.caption("Si no se dispone de una medición, marca «No disponible»: el modelo la "
                   "imputa con la mediana del entrenamiento.")
        valores = {}
        cols = st.columns(2)
        for i, (var, (etiqueta, mn, mx, defecto, paso, puede_faltar)) in enumerate(CAMPOS.items()):
            with cols[i % 2]:
                faltante = False
                if puede_faltar:
                    faltante = st.checkbox(f"{etiqueta}: no disponible", key=f"nd_{var}")
                v = st.number_input(etiqueta, min_value=mn, max_value=mx, value=defecto,
                                    step=paso, key=var, disabled=faltante,
                                    help=f"Rango válido: {mn} a {mx}")
                valores[var] = np.nan if faltante else float(v)
        fila = pd.DataFrame([valores], columns=columnas)
        real = None
    else:
        idx = st.selectbox("Índice del registro (conjunto de prueba)", list(X_test.index),
                           index=list(X_test.index).index(44) if 44 in X_test.index else 0)
        fila = X_test.loc[[idx], columnas]
        real = int(y_test.loc[idx])
        st.dataframe(fila)

    if st.button("Predecir", type="primary"):
        fila = fila.copy()
        fila[COLS_CERO_NAN] = fila[COLS_CERO_NAN].replace(0, np.nan)
        clase = int(modelo.predict(fila)[0])
        prob = float(modelo.predict_proba(fila)[0, 1])
        if clase == 1:
            st.error(f"Resultado: **riesgo alto de diabetes** (clase 1)")
        else:
            st.success(f"Resultado: **riesgo bajo de diabetes** (clase 0)")
        st.metric("Probabilidad de diabetes", f"{prob:.4f}")
        st.progress(min(max(prob, 0.0), 1.0))
        if real is not None:
            st.info(f"Valor real en el conjunto de prueba: **{real}**")
        st.caption("Umbral de decisión: 0.5. Prototipo académico, no es un diagnóstico.")

# ------------------------------------------------------------------ Métricas
with tab4:
    st.header("Desempeño en el conjunto de prueba")
    st.caption(f"{len(X_test)} registros (división estratificada 80/20, semilla {SEED}). "
               "El modelo se carga del archivo; no se reentrena.")
    pred = modelo.predict(X_test[columnas])
    proba = modelo.predict_proba(X_test[columnas])[:, 1]
    m = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision (diabética)", "Recall (diabética)", "F1 (diabética)", "AUC-ROC"],
        "Valor": [accuracy_score(y_test, pred), precision_score(y_test, pred),
                  recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, proba)]})
    m["Valor"] = m["Valor"].round(3)
    c1, c2 = st.columns(2)
    with c1:
        st.table(m.set_index("Métrica"))
    with c2:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                               display_labels=["No diab.", "Diab."]).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
    st.subheader("Limitaciones")
    st.markdown("""
- La evaluación se basa en solo 154 pacientes de prueba; los resultados pueden variar con otra partición.
- El recall (≈0.70) deja pacientes diabéticas sin detectar (falsos negativos).
- No se optimizaron hiperparámetros ni el umbral de decisión (0.5).
- La glucosa forma parte de los criterios diagnósticos: el problema se parece más a apoyo al diagnóstico que a predicción temprana.
- El dataset solo incluye mujeres de origen Pima, lo que limita la generalización.
- Los ceros imposibles se tratan como datos faltantes e imputados con la mediana (línea base).
""")
