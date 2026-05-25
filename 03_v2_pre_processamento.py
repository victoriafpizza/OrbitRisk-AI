import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv("dataset_orbitrisk_feature_engineering.csv")

# Remover colunas textuais pouco úteis para o modelo
df = df.drop(columns=["data_evento", "nome_objeto"])

# Converter booleano para inteiro
df["potencialmente_perigoso"] = df["potencialmente_perigoso"].astype(int)

# Codificar variáveis categóricas
encoder_impacto = LabelEncoder()
df["impacto_comunicacao"] = encoder_impacto.fit_transform(df["impacto_comunicacao"])

encoder_orbita = LabelEncoder()
df["faixa_orbital"] = encoder_orbita.fit_transform(df["faixa_orbital"])

encoder_risco = LabelEncoder()
df["risco_operacional"] = encoder_risco.fit_transform(df["risco_operacional"])

X = df.drop(columns=["risco_operacional"])
y = df["risco_operacional"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Normalização dos dados
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Salvar arquivos
X_train_scaled.to_csv("X_train.csv", index=False)
X_test_scaled.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Salvar objetos importantes para o app
joblib.dump(scaler, "scaler_orbitrisk.pkl")
joblib.dump(encoder_impacto, "encoder_impacto.pkl")
joblib.dump(encoder_orbita, "encoder_orbita.pkl")
joblib.dump(encoder_risco, "encoder_risco.pkl")

print("Pré-processamento V2 finalizado com sucesso!")
print("Quantidade de linhas:", df.shape[0])
print("Quantidade de colunas:", df.shape[1])
print("Colunas usadas no modelo:")
print(X.columns.tolist())
print("Classes do risco:", encoder_risco.classes_)