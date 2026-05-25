import pandas as pd
import numpy as np

df = pd.read_csv("dados_nasa_neo.csv")

# Criar colunas sintéticas complementares para completar o dataset
np.random.seed(42)

df["altitude_orbital_km"] = np.random.randint(300, 36000, size=len(df))
df["indice_kp_geomagnetico"] = np.random.uniform(0, 9, size=len(df)).round(2)
df["intensidade_evento_solar"] = np.random.uniform(0, 100, size=len(df)).round(2)
df["impacto_comunicacao"] = np.random.choice(["baixo", "medio", "alto"], size=len(df))

# Criar variável alvo: risco_operacional
def definir_risco(row):
    if (
        row["distancia_minima_km"] < 1000000
        or row["velocidade_km_s"] > 25
        or row["indice_kp_geomagnetico"] > 6
        or row["intensidade_evento_solar"] > 75
    ):
        return "alto"
    elif (
        row["distancia_minima_km"] < 5000000
        or row["velocidade_km_s"] > 15
        or row["indice_kp_geomagnetico"] > 4
        or row["intensidade_evento_solar"] > 50
    ):
        return "medio"
    else:
        return "baixo"

df["risco_operacional"] = df.apply(definir_risco, axis=1)

# Expandir até ter pelo menos 1000 linhas
df_final = pd.concat([df] * ((1000 // len(df)) + 1), ignore_index=True)
df_final = df_final.head(1000)

df_final.to_csv("dataset_orbitrisk_ai.csv", index=False)

print("Dataset final criado com sucesso!")
print("Linhas:", df_final.shape[0])
print("Colunas:", df_final.shape[1])
print(df_final.head())