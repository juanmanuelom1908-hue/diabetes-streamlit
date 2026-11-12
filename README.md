# Predicción de diabetes tipo 2 con Streamlit

Aplicación web que integra el modelo de clasificación del proyecto de Minería de Datos (tercer corte).

- **Integrante:** Juan Ortega
- **Curso:** Minería de Datos, Universidad de la Costa (CUC)
- **Aplicación:** <PEGAR_URL_STREAMLIT>
- **Datos:** Pima Indians Diabetes Database (NIDDK; Smith et al., 1988), 768 registros.

> **Prototipo académico.** No sustituye el diagnóstico ni la decisión de un profesional de la salud.

## Modelo
Pipeline de scikit-learn guardado con `joblib` (`modelo_diabetes.joblib`): imputación por mediana, escalamiento estándar y regresión logística con pesos de clase balanceados. La app lo carga; no reentrena. Los ceros imposibles de Glucose, BloodPressure, SkinThickness, Insulin y BMI se tratan como datos faltantes (`NaN`) antes de predecir.

Resultados en prueba (n = 154): accuracy 0.734, precision 0.603, recall 0.704, F1 0.65, AUC-ROC 0.813.

## Archivos
- `app.py`: aplicación Streamlit
- `modelo_diabetes.joblib`, `columnas.joblib`: pipeline y orden de columnas
- `diabetes.csv`: datos (para el EDA y las métricas)
- `requirements.txt`
- `Avance3_Proyecto_Mineria_Datos.ipynb`: notebook

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pruebas realizadas
| Caso | Resultado |
|---|---|
| Normal (valores por defecto) | Probabilidad 0.3043, clase 0 |
| Entrada inválida (fuera de rango) | El formulario no permite valores fuera del rango definido |
| Registro conocido del conjunto de prueba (índice 44) | Streamlit: 0.7332, clase 1. Notebook: 0.7332, clase 1 |

## Uso de inteligencia artificial
| Elemento | Información |
|---|---|
| Herramienta | Claude (Anthropic) |
| Propósito | Revisar el pipeline del notebook, guardarlo con joblib y construir `app.py`, `requirements.txt` y este README |
| Prompt principal | «Hola analiza este archivo» (guía de la actividad) y «Hazlo tú» sobre el pipeline y la app |
| Verificación | Se reprodujeron las métricas del informe del segundo corte y se comparó la probabilidad del registro 44 entre notebook y app |
| Problemas encontrados | <COMPLETAR: p. ej., el notebook convierte los ceros a NaN fuera del pipeline, por lo que la app debe repetir ese paso; la IA usó una copia pública del dataset, no el archivo propio> |
| Ajustes realizados | <COMPLETAR> |

Declaro que revisé el código entregado y comprendo su funcionamiento.
