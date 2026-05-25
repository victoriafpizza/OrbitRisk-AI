import requests
import pandas as pd

API_KEY = "DEMO_KEY"

url = "https://api.nasa.gov/neo/rest/v1/feed"

params = {
    "start_date": "2024-01-01",
    "end_date": "2024-01-07",
    "api_key": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

linhas = []

for data_evento, objetos in data["near_earth_objects"].items():
    for obj in objetos:
        linhas.append({
            "data_evento": data_evento,
            "nome_objeto": obj["name"],
            "magnitude_absoluta": obj["absolute_magnitude_h"],
            "diametro_min_m": obj["estimated_diameter"]["meters"]["estimated_diameter_min"],
            "diametro_max_m": obj["estimated_diameter"]["meters"]["estimated_diameter_max"],
            "velocidade_km_s": float(obj["close_approach_data"][0]["relative_velocity"]["kilometers_per_second"]),
            "distancia_minima_km": float(obj["close_approach_data"][0]["miss_distance"]["kilometers"]),
            "potencialmente_perigoso": obj["is_potentially_hazardous_asteroid"]
        })

df = pd.DataFrame(linhas)

df.to_csv("dados_nasa_neo.csv", index=False)

print("Arquivo dados_nasa_neo.csv criado com sucesso!")
print(df.head())