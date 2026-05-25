import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

resultados = []

melhor_modelo = None
melhor_nome = ""
melhor_f1 = 0

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    resultados.append({
        "modelo": nome,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\n==============================")
    print(f"Modelo: {nome}")
    print("==============================")
    print(classification_report(y_test, y_pred))

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo = modelo
        melhor_nome = nome

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_modelos.csv", index=False)

joblib.dump(melhor_modelo, "melhor_modelo_orbitrisk.pkl")

print("\nResultados salvos em resultados_modelos.csv")
print("Melhor modelo:", melhor_nome)
print("F1-score:", melhor_f1)
print("Modelo salvo como melhor_modelo_orbitrisk.pkl")