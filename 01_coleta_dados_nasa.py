import requests
import pandas as pd

API_KEY = "DEMO_KEY"

url = "https://api.nasa.gov/neo/rest/v1/feed"

params = {
    "start_date": "2024-01-01",
    "end_date": "2024-01-07",
    "api_key": API_KEY
}

try:
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Erro na API NASA")
        print("Status code:", response.status_code)
        print(response.text)
        exit()

    data = response.json()

    linhas = []

    near_objects = data.get("near_earth_objects", {})

    for data_evento, objetos in near_objects.items():
        for obj in objetos:

            close_data = obj.get("close_approach_data", [])

            if not close_data:
                continue

            linhas.append({
                "data_evento": data_evento,
                "nome_objeto": obj.get("name", "desconhecido"),
                "magnitude_absoluta": obj.get("absolute_magnitude_h", 0),

                "diametro_min_m":
                    obj["estimated_diameter"]["meters"]
                    .get("estimated_diameter_min", 0),

                "diametro_max_m":
                    obj["estimated_diameter"]["meters"]
                    .get("estimated_diameter_max", 0),

                "velocidade_km_s":
                    float(
                        close_data[0]["relative_velocity"]
                        .get("kilometers_per_second", 0)
                    ),

                "distancia_minima_km":
                    float(
                        close_data[0]["miss_distance"]
                        .get("kilometers", 0)
                    ),

                "potencialmente_perigoso":
                    obj.get(
                        "is_potentially_hazardous_asteroid",
                        False
                    )
            })

    df = pd.DataFrame(linhas)

    if len(df) == 0:
        print("Nenhum dado encontrado.")
    else:
        df.to_csv("dados_nasa_neo.csv", index=False)

        print("\nColeta concluída!")
        print(f"Linhas coletadas: {len(df)}")
        print("Arquivo salvo: dados_nasa_neo.csv")
        print(df.head())

except Exception as e:
    print("Erro:", e)