import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("melhor_modelo_orbitrisk.pkl")
scaler = joblib.load("scaler_orbitrisk.pkl")
encoder_risco = joblib.load("encoder_risco.pkl")

st.set_page_config(
    page_title="OrbitRisk AI",
    page_icon="🛰️",
    layout="centered"
)

st.title("OrbitRisk AI")
st.subheader("Previsão de risco operacional em missões espaciais")

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
    ["alto", "baixo", "medio"]
)

def classificar_orbita(altitude):
    if altitude < 2000:
        return 1  # LEO
    elif altitude < 35786:
        return 2  # MEO
    else:
        return 0  # GEO

mapa_perigoso = {
    "Não": 0,
    "Sim": 1
}

mapa_impacto = {
    "alto": 0,
    "baixo": 1,
    "medio": 2
}

razao_diametro = diametro_max_m / (diametro_min_m + 1)

energia_aproximada = diametro_max_m * (velocidade_km_s ** 2)

indice_ameaca_orbital = (
    (velocidade_km_s * 0.4) +
    (indice_kp_geomagnetico * 0.3) +
    (intensidade_evento_solar * 0.2)
)

distancia_milhoes_km = distancia_minima_km / 1_000_000

faixa_orbital = classificar_orbita(altitude_orbital_km)

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
    "impacto_comunicacao": mapa_impacto[impacto_comunicacao],
    "razao_diametro": razao_diametro,
    "energia_aproximada": energia_aproximada,
    "indice_ameaca_orbital": indice_ameaca_orbital,
    "distancia_milhoes_km": distancia_milhoes_km,
    "faixa_orbital": faixa_orbital
}])

if st.button("Prever risco"):
    dados_escalados = scaler.transform(dados)

    predicao = modelo.predict(dados_escalados)[0]
    probabilidades = modelo.predict_proba(dados_escalados)[0]

    risco = encoder_risco.inverse_transform([predicao])[0]

    if risco == "alto":
        st.error("Risco operacional previsto: ALTO")
        st.warning("Recomendação: revisar a janela de operação e monitorar clima espacial.")
    elif risco == "medio":
        st.warning("Risco operacional previsto: MÉDIO")
        st.info("Recomendação: manter monitoramento contínuo antes da operação.")
    else:
        st.success("Risco operacional previsto: BAIXO")
        st.info("Recomendação: operação favorável com monitoramento padrão.")

    df_prob = pd.DataFrame({
        "Classe": encoder_risco.classes_,
        "Probabilidade": probabilidades
    })

    st.write("Probabilidades por classe:")
    st.dataframe(df_prob)