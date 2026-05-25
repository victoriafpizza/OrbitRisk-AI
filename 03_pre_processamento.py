import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset_orbitrisk_ai.csv")

# Remover colunas textuais que não ajudam diretamente no modelo
df = df.drop(columns=["data_evento", "nome_objeto"])

# Converter valores booleanos para inteiro
df["potencialmente_perigoso"] = df["potencialmente_perigoso"].astype(int)

# Codificar coluna categórica
encoder_impacto = LabelEncoder()
df["impacto_comunicacao"] = encoder_impacto.fit_transform(df["impacto_comunicacao"])

# Codificar variável alvo
encoder_risco = LabelEncoder()
df["risco_operacional"] = encoder_risco.fit_transform(df["risco_operacional"])

X = df.drop(columns=["risco_operacional"])
y = df["risco_operacional"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Pré-processamento finalizado com sucesso!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Classes:", encoder_risco.classes_)