# OrbitRisk AI

Sistema inteligente de previsão de risco operacional para missões espaciais utilizando Inteligência Artificial e Machine Learning.
Desenvolvido por: Victoria Franceschini Pizza | RM550609

## Objetivo do Projeto

O OrbitRisk AI foi desenvolvido para prever o nível de risco operacional de missões espaciais e satélites com base em dados relacionados a objetos próximos da Terra (Near Earth Objects - NEO), clima espacial e variáveis orbitais.

O sistema utiliza técnicas de Machine Learning para classificar o risco operacional em três níveis:

- Baixo
- Médio
- Alto

O objetivo é auxiliar análises preventivas e apoiar decisões operacionais relacionadas à Economia Espacial.

---

## Problema de Negócio

Na Economia Espacial, satélites e missões podem sofrer impactos devido a:

- Aproximação de objetos espaciais
- Asteroides potencialmente perigosos
- Tempestades geomagnéticas
- Eventos solares
- Interferências de comunicação

O projeto propõe uma solução preditiva baseada em IA capaz de antecipar cenários de risco operacional.

---

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- SHAP
- Streamlit
- Matplotlib
- Joblib
- Requests API

---

## Fonte dos Dados

Os dados foram obtidos através da API oficial da NASA:

:contentReference[oaicite:0]{index=0}

Mais especificamente, foram utilizados dados de objetos próximos da Terra (Near Earth Objects - NEO).

Também foram gerados dados sintéticos para complementar o dataset e atingir os requisitos mínimos do projeto.

---

## Estrutura do Projeto

```text
OrbitRisk-AI/
│
├── 01_coleta_dados_nasa.py
├── 02_gerar_dataset_completo.py
├── 03_v2_pre_processamento.py
├── 04_treinamento_modelos.py
├── 05_interpretabilidade_shap.py
├── 06_app_streamlit.py
├── 07_engenharia_atributos.py
│
├── dados_nasa_neo.csv
├── dataset_orbitrisk_ai.csv
├── dataset_orbitrisk_feature_engineering.csv
│
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── melhor_modelo_orbitrisk.pkl
├── scaler_orbitrisk.pkl
│
├── resultados_modelos.csv
│
├── shap_summary_classe_0.png
├── shap_summary_classe_1.png
├── shap_summary_classe_2.png
│
├── requirements.txt
└── README.md
```

---

## Pipeline de Machine Learning

O projeto segue um pipeline completo de Machine Learning:

### 1. Coleta de Dados

Os dados são coletados diretamente da API da NASA utilizando Python e Requests.

### 2. Geração do Dataset

Foi criado um dataset híbrido contendo:

- Dados reais da NASA
- Dados sintéticos
- Mais de 1.000 registros
- Mais de 10 variáveis

### 3. Pré-processamento

Etapas realizadas:

- Limpeza dos dados
- Tratamento de variáveis categóricas
- Codificação com LabelEncoder
- Normalização com StandardScaler
- Separação treino/teste

### 4. Engenharia de Atributos

Foram criadas variáveis derivadas para aumentar a capacidade preditiva do modelo:

- Razão entre diâmetros
- Energia aproximada
- Índice de ameaça orbital
- Distância em milhões de quilômetros
- Faixa orbital

### 5. Modelos Testados

Foram avaliados dois algoritmos de Machine Learning:

#### Random Forest Classifier
Modelo baseado em árvores de decisão combinadas.

#### Gradient Boosting Classifier
Modelo baseado em boosting sequencial para aumento de precisão.

### 6. Avaliação dos Modelos

Foram utilizadas as seguintes métricas:

- Accuracy
- Precision
- Recall
- F1-Score

O melhor modelo foi salvo automaticamente para uso no sistema final.

---

## Interpretabilidade do Modelo com SHAP

Foi utilizada a biblioteca SHAP (SHapley Additive exPlanations) para interpretar o comportamento do modelo.

O SHAP permite identificar:

- Quais variáveis possuem maior influência
- Como o modelo toma decisões
- Impacto de cada feature na classificação do risco

Gráficos SHAP foram gerados automaticamente durante o treinamento.

---

## Aplicação Web

O projeto possui uma aplicação interativa desenvolvida com Streamlit.

O usuário pode informar parâmetros da missão espacial e o sistema retorna:

- Classificação do risco operacional
- Probabilidade da previsão
- Recomendação operacional

---

## Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone LINK_DO_SEU_REPOSITORIO
```

### 2. Entre na pasta do projeto

```bash
cd OrbitRisk-AI
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline

```bash
python 01_coleta_dados_nasa.py
python 02_gerar_dataset_completo.py
python 07_engenharia_atributos.py
python 03_v2_pre_processamento.py
python 04_treinamento_modelos.py
python 05_interpretabilidade_shap.py
```

### 5. Executar a aplicação

```bash
python -m streamlit run 06_app_streamlit.py
```

---

## Resultados

O sistema foi capaz de classificar riscos operacionais utilizando dados espaciais e variáveis orbitais com modelos supervisionados de Machine Learning.

Além disso, a interpretabilidade do modelo permitiu identificar quais fatores exercem maior influência nas previsões.

---

## Autores

Projeto acadêmico desenvolvido por Victoria Franceschini Pizza para a disciplina de Generative AI and Engineering aplicado à Economia Espacial.
https://orbitrisk-ai-zwwffadhpwcgpustrc5osc.streamlit.app/