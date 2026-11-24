from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
COLS_CERO_IMPOSIBLE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Riesgo de diabetes (proyecto académico)", page_icon="🩺")


@st.cache_resource
def cargar():
    modelo = joblib.load(BASE / "modelo_diabetes.joblib")
    columnas = joblib.load(BASE / "columnas.joblib")
    umbral = joblib.load(BASE / "umbral.joblib")
    return modelo, columnas, umbral


def preparar_entrada(df, columnas):
    """Misma transformación del notebook: ceros imposibles -> NaN y orden de columnas."""
    d = df[list(columnas)].copy()
    d[COLS_CERO_IMPOSIBLE] = d[COLS_CERO_IMPOSIBLE].replace(0, np.nan)
    return d


modelo, columnas, umbral = cargar()

st.title("🩺 Clasificación de riesgo de diabetes tipo 2")
st.warning(
    "Herramienta **académica** (proyecto de Minería de Datos, CUC). Sus resultados no son un "
    "diagnóstico y **no sustituyen una valoración profesional de salud**."
)

st.subheader("Datos de la paciente")
c1, c2 = st.columns(2)
with c1:
    embarazos = st.number_input("Embarazos previos", 0, 20, 1)
    edad = st.number_input("Edad (años)", 21, 100, 30)
    dpf = st.number_input("Función de antecedentes familiares (DiabetesPedigreeFunction)", 0.05, 2.5, 0.35, 0.001, format="%.3f")
with c2:
    glucosa = st.number_input("Glucosa a 2 h (mg/dL)", 40, 300, 120)
    presion = st.number_input("Presión diastólica (mm Hg)", 20, 140, 70)
    imc = st.number_input("IMC (kg/m²)", 15.0, 70.0, 28.0, 0.1)

st.caption("Si una medición no está disponible, márcala: el modelo la trata como dato faltante.")
d1, d2 = st.columns(2)
with d1:
    sin_piel = st.checkbox("Grosor de pliegue cutáneo no disponible")
    piel = 0 if sin_piel else st.number_input("Pliegue cutáneo del tríceps (mm)", 5, 100, 25)
with d2:
    sin_insulina = st.checkbox("Insulina no disponible")
    insulina = 0 if sin_insulina else st.number_input("Insulina a 2 h (mu U/ml)", 10, 900, 80)

if st.button("Calcular"):
    registro = pd.DataFrame([{
        "Pregnancies": embarazos, "Glucose": glucosa, "BloodPressure": presion,
        "SkinThickness": piel, "Insulin": insulina, "BMI": imc,
        "DiabetesPedigreeFunction": dpf, "Age": edad,
    }])
    prob = float(modelo.predict_proba(preparar_entrada(registro, columnas))[0, 1])
    clase = int(prob >= umbral)

    st.metric("Probabilidad estimada por el modelo", f"{prob:.1%}")
    if clase == 1:
        st.error(f"El modelo clasifica este perfil como **de riesgo** (probabilidad ≥ umbral {umbral:.2f}).")
    else:
        st.success(f"El modelo clasifica este perfil como **de bajo riesgo** (probabilidad < umbral {umbral:.2f}).")
    st.caption("Resultado académico basado en el conjunto Pima (mujeres de ≥ 21 años); no generaliza a otras poblaciones.")
