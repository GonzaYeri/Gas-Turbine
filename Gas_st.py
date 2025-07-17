import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 🚀 Cargar modelos
def load_components():
    model_tey = joblib.load('best_model_tey.joblib')
    model_nox = joblib.load('best_model_nox.joblib')
    model_co  = joblib.load('best_model_co.joblib')
    return model_tey, model_nox, model_co

model_tey, model_nox, model_co = load_components()

# 🎯 Título principal
st.title("🌍 Análisis completo de TEY / NOX / CO – Predicción, Tendencias y Rendimiento")

# 🗓️ Selección de año
selected_year = st.number_input("🗓️ Año de análisis:", min_value=1900, max_value=2100, value=2014, step=1)

# 📤 Subida de dataset
uploaded_file = st.file_uploader("📁 Subí el dataset crudo (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Leer archivo
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("👀 Vista previa del dataset")
        st.dataframe(df.head())

        # 🎯 Validar columnas necesarias por modelo
        features_tey = model_tey.feature_names_in_
        features_nox = model_nox.feature_names_in_
        features_co  = model_co.feature_names_in_

        missing = {
            "TEY": set(features_tey) - set(df.columns),
            "NOX": set(features_nox) - set(df.columns),
            "CO":  set(features_co)  - set(df.columns)
        }

        if any(missing.values()):
            st.error(f"❌ Columnas faltantes:\nTEY → {missing['TEY']}\nNOX → {missing['NOX']}\nCO → {missing['CO']}")
        else:
            # 📆 Generar columna Mes
            df["Date"] = pd.date_range(start=f"{selected_year}-01-01", periods=len(df), freq="H")
            df["Mes"] = df["Date"].dt.to_period("M").astype(str)

            # 🔍 Predicción por modelo
            df["TEY_Pred"] = model_tey.predict(df[features_tey])
            df["NOX_Pred"] = model_nox.predict(df[features_nox])
            df["CO_Pred"]  = model_co.predict(df[features_co])

            # 📊 Agrupación mensual
            mensual = df.groupby("Mes").agg({
                "TEY": "mean", "TEY_Pred": "mean",
                "NOX": "mean", "NOX_Pred": "mean",
                "CO": "mean",  "CO_Pred":  "mean"
            }).reset_index()
            mensual["Mes"] = pd.to_datetime(mensual["Mes"], format="%Y-%m")

            # 📉 Cálculo de diferencia porcentual mensual
            mensual["TEY_Diff_%"] = (mensual["TEY"] - mensual["TEY_Pred"]) / mensual["TEY_Pred"] * 100
            mensual["NOX_Diff_%"] = (mensual["NOX"] - mensual["NOX_Pred"]) / mensual["NOX_Pred"] * 100
            mensual["CO_Diff_%"]  = (mensual["CO"]  - mensual["CO_Pred"])  / mensual["CO_Pred"]  * 100

            # 📆 Cálculo de promedios anuales
            real_tey = df['TEY'].mean()
            pred_tey = df['TEY_Pred'].mean()
            real_nox = df['NOX'].mean()
            pred_nox = df['NOX_Pred'].mean()
            real_co  = df['CO'].mean()
            pred_co  = df['CO_Pred'].mean()

            # 🧠 Mensaje interpretativo anual
            def generar_mensaje(real, pred, nombre):
                diff = real - pred
                pct = (diff / pred) * 100

                if nombre == "TEY":
                    if diff >= 0:
                        return f"🔴 TEY real: {real:.2f}\n🟢 Predicho: {pred:.2f}\n✅ El rendimiento fue {pct:.2f}% superior al estimado."
                    else:
                        return f"🔴 TEY real: {real:.2f}\n🟢 Predicho: {pred:.2f}\n⚠️ El rendimiento fue {abs(pct):.2f}% inferior al estimado."
                else:
                    if diff > 0:
                        return f"🔴 {nombre} real: {real:.2f}\n🟢 Predicho: {pred:.2f}\n⚠️ Emisiones fueron {pct:.2f}% superiores a lo esperado."
                    else:
                        return f"🔴 {nombre} real: {real:.2f}\n🟢 Predicho: {pred:.2f}\n✅ Emisiones fueron {abs(pct):.2f}% inferiores a lo estimado."

            # 🪄 Mostrar resultados anuales
            st.markdown("### 📊 Resultados Anuales")
            st.markdown(generar_mensaje(real_tey, pred_tey, "TEY"))
            st.markdown(generar_mensaje(real_nox, pred_nox, "NOX"))
            st.markdown(generar_mensaje(real_co, pred_co, "CO"))

            # 📈 Función para gráfico mensual
            def plot_variable(df, real, pred, label):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df["Mes"], df[real], label=f"{label} Real", color="red", marker="o")
                ax.plot(df["Mes"], df[pred], label=f"{label} Predicho", color="green", linestyle="--", marker="x")
                ax.set_title(f"Tendencia mensual de {label}")
                ax.set_xlabel("Mes")
                ax.set_ylabel(f"{label} promedio")
                ax.legend()
                ax.grid(True)
                return fig

            def plot_diferencia(df, diff_col, label):
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["Mes"], df[diff_col], color="blue", marker="o")
                ax.axhline(0, color="gray", linestyle="--")
                ax.set_title(f"Desvío mensual: {label} real vs. predicho")
                ax.set_ylabel("Diferencia %")
                ax.set_xlabel("Mes")
                ax.grid(True)
                return fig

            # 📈 Mostrar gráficos por variable
            st.subheader("📈 TEY – Real vs. Predicho")
            st.pyplot(plot_variable(mensual, "TEY", "TEY_Pred", "TEY"))

            st.subheader("📈 NOX – Real vs. Predicho")
            st.pyplot(plot_variable(mensual, "NOX", "NOX_Pred", "NOX"))

            st.subheader("📈 CO – Real vs. Predicho")
            st.pyplot(plot_variable(mensual, "CO", "CO_Pred", "CO"))

    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")