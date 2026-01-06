# ============================================================
# APP STREAMLIT - RETAILMAX
# Dashboard Interactivo de Predicción de Ventas
# Autor: Elda Serrano
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="RetailMax - Predictor de Ventas",
    layout="wide"
)

st.title("📊 RetailMax - Predictor de Ventas")
st.markdown("Sistema interactivo de predicción de ventas semanales")

# ------------------------------------------------------------
# CARGA DE DATOS DEMO
# ------------------------------------------------------------
@st.cache_data
def cargar_datos():
    np.random.seed(42)
    tiendas = [f"Tienda_{i}" for i in range(1, 21)]
    fechas = pd.date_range("2024-01-01", periods=12, freq="W")

    data = []
    for tienda in tiendas:
        for fecha in fechas:
            data.append({
                "tienda_id": tienda,
                "fecha": fecha,
                "ventas_semanales": np.random.normal(15000, 3000),
                "promocion_activa": np.random.choice([0,1]),
                "inventario_inicial": np.random.normal(50000, 10000),
                "temperatura_promedio": np.random.normal(20, 8),
                "año": fecha.year,
                "mes": fecha.month,
                "semana": fecha.isocalendar().week,
                "dia_semana": fecha.dayofweek
            })
    return pd.DataFrame(data)

df = cargar_datos()

# ------------------------------------------------------------
# ENTRENAR MODELO DEMO
# ------------------------------------------------------------
@st.cache_resource
def entrenar_modelo(df):
    features = [
        "promocion_activa", "inventario_inicial", "temperatura_promedio",
        "año", "mes", "semana", "dia_semana"
    ]
    X = df[features]
    y = df["ventas_semanales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, features

modelo, features = entrenar_modelo(df)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuración de Predicción")

tienda = st.sidebar.selectbox("Tienda", df["tienda_id"].unique())
fecha = st.sidebar.date_input("Fecha", datetime.now() + timedelta(days=7))
promocion = st.sidebar.radio("Promoción activa", [0,1], format_func=lambda x: "Sí" if x==1 else "No")
inventario = st.sidebar.slider("Inventario inicial", 20000, 80000, 50000, step=1000)
temperatura = st.sidebar.slider("Temperatura promedio", -5, 40, 20)

predecir = st.sidebar.button("🔮 Predecir")

# ------------------------------------------------------------
# PREDICCIÓN
# ------------------------------------------------------------
if predecir:
    entrada = pd.DataFrame([{
        "promocion_activa": promocion,
        "inventario_inicial": inventario,
        "temperatura_promedio": temperatura,
        "año": fecha.year,
        "mes": fecha.month,
        "semana": fecha.isocalendar().week,
        "dia_semana": fecha.dayofweek
    }])

    pred = modelo.predict(entrada)[0]

    st.subheader("📈 Resultado de Predicción")
    st.metric("Ventas estimadas", f"${pred:,.0f}")

    # Comparación histórica
    historico = df[df["tienda_id"] == tienda]

    fig = px.line(
        historico,
        x="fecha",
        y="ventas_semanales",
        title=f"Ventas históricas - {tienda}"
    )

    fig.add_scatter(
        x=[fecha],
        y=[pred],
        mode="markers",
        marker=dict(color="red", size=12),
        name="Predicción"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("📊 Dashboard General")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tiendas", df["tienda_id"].nunique())
    col2.metric("Semanas", df["fecha"].nunique())
    col3.metric("Promedio Ventas", f"${df['ventas_semanales'].mean():,.0f}")
