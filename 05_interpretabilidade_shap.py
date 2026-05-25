import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

modelo = joblib.load("melhor_modelo_orbitrisk.pkl")

# Usar TreeExplainer, ideal para Random Forest e Gradient Boosting
explainer = shap.TreeExplainer(modelo)

shap_values = explainer.shap_values(X_test)

# Caso o modelo seja multiclasse, pegamos uma classe por vez
if isinstance(shap_values, list):
    for i, valores_classe in enumerate(shap_values):
        plt.figure()
        shap.summary_plot(
            valores_classe,
            X_test,
            show=False
        )
        plt.savefig(f"shap_summary_classe_{i}.png", bbox_inches="tight")
        plt.close()

else:
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )
    plt.savefig("shap_summary_plot.png", bbox_inches="tight")
    plt.close()

print("Análise SHAP finalizada com sucesso!")
print("Gráficos SHAP salvos na pasta do projeto.")