import pandas as pd
import numpy as np

df = pd.read_csv("/data/dataset_orbitrisk_ai.csv")

# 1. Razão entre diâmetros
df["razao_diametro"] = (
    df["diametro_max_m"] /
    (df["diametro_min_m"] + 1)
)

# 2. Energia cinética simplificada
df["energia_aproximada"] = (
    df["diametro_max_m"] *
    (df["velocidade_km_s"] ** 2)
)

# 3. Índice de ameaça orbital
df["indice_ameaca_orbital"] = (
    (df["velocidade_km_s"] * 0.4) +
    (df["indice_kp_geomagnetico"] * 0.3) +
    (df["intensidade_evento_solar"] * 0.2)
)

# 4. Distância em milhões de km
df["distancia_milhoes_km"] = (
    df["distancia_minima_km"] / 1_000_000
)

# 5. Faixa orbital
def classificar_orbita(altitude):
    if altitude < 2000:
        return "LEO"
    elif altitude < 35786:
        return "MEO"
    else:
        return "GEO"

df["faixa_orbital"] = df["altitude_orbital_km"].apply(classificar_orbita)

# Salvar novo dataset
df.to_csv("dataset_orbitrisk_feature_engineering.csv", index=False)

print("Feature Engineering concluído!")
print("Novas colunas criadas:")
print(df.columns.tolist())