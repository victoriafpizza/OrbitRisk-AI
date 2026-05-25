import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("melhor_modelo_orbitrisk.pkl")

st.set_page_config(
    page_title="OrbitRisk AI",
    page_icon="🛰️",
    layout="centered"
)

st.title("OrbitRisk AI")
st.subheader("Previsão de risco operacional em missões espaciais")

st.write(
    "Informe os dados da operação orbital para prever o nível de risco."
)

magnitude_absoluta = st.number_input("Magnitude absoluta", value=22.0)
diametro_min_m = st.number_input("Diâmetro mínimo estimado (m)", value=50.0)
diametro_max_m = st.number_input("Diâmetro máximo estimado (m)", value=120.0)
velocidade_km_s = st.number_input("Velocidade do objeto (km/s)", value=15.0)
distancia_minima_km = st.number_input("Distância mínima do objeto (km)", value=5000000.0)
potencialmente_perigoso = st.selectbox(
    "Objeto potencialmente perigoso?",
    ["Não", "Sim"]
)
altitude_orbital_km = st.number_input("Altitude orbital (km)", value=550.0)
indice_kp_geomagnetico = st.slider("Índice KP geomagnético", 0.0, 9.0, 3.0)
intensidade_evento_solar = st.slider("Intensidade do evento solar", 0.0, 100.0, 30.0)
impacto_comunicacao = st.selectbox(
    "Impacto na comunicação",
    ["baixo", "medio", "alto"]
)

mapa_perigoso = {
    "Não": 0,
    "Sim": 1
}

mapa_impacto = {
    "alto": 0,
    "baixo": 1,
    "medio": 2
}

dados = pd.DataFrame([{
    "magnitude_absoluta": magnitude_absoluta,
    "diametro_min_m": diametro_min_m,
    "diametro_max_m": diametro_max_m,
    "velocidade_km_s": velocidade_km_s,
    "distancia_minima_km": distancia_minima_km,
    "potencialmente_perigoso": mapa_perigoso[potencialmente_perigoso],
    "altitude_orbital_km": altitude_orbital_km,
    "indice_kp_geomagnetico": indice_kp_geomagnetico,
    "intensidade_evento_solar": intensidade_evento_solar,
    "impacto_comunicacao": mapa_impacto[impacto_comunicacao]
}])

mapa_risco = {
    0: "Alto",
    1: "Baixo",
    2: "Médio"
}

if st.button("Prever risco"):
    predicao = modelo.predict(dados)[0]
    probabilidades = modelo.predict_proba(dados)[0]

    risco = mapa_risco[predicao]

    st.success(f"Risco operacional previsto: {risco}")

    st.write("Probabilidades por classe:")

    df_prob = pd.DataFrame({
        "Classe": ["Alto", "Baixo", "Médio"],
        "Probabilidade": probabilidades
    })

    st.dataframe(df_prob)

    if risco == "Alto":
        st.warning(
            "Recomendação: revisar a janela de operação, monitorar clima espacial e avaliar manobra preventiva."
        )
    elif risco == "Médio":
        st.info(
            "Recomendação: manter monitoramento contínuo e validar condições antes da operação."
        )
    else:
        st.success(
            "Recomendação: operação em condição favorável, com monitoramento padrão."
        )