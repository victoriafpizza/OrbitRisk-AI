import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

modelo = joblib.load("melhor_modelo_orbitrisk.pkl")

# Criar explicador SHAP
explainer = shap.Explainer(modelo, X_train)

# Calcular valores SHAP
shap_values = explainer(X_test)

# Gráfico geral de importância das variáveis
shap.plots.bar(shap_values, show=False)
plt.savefig("shap_importancia_variaveis.png", bbox_inches="tight")
plt.close()

# Gráfico detalhado
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_plot.png", bbox_inches="tight")
plt.close()

print("Análise SHAP finalizada com sucesso!")
print("Arquivos gerados:")
print("- shap_importancia_variaveis.png")
print("- shap_summary_plot.png")