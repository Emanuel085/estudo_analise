#importar as bibliotecas e de análises avançadas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

#configuraçao de estilo
sns.set(style="whitegrid")

#carregar o csv
df = pd.read_csv("Tabela1.csv", sep=";", engine="python")

#Corrigir salário (trocar vírgula por ponto e converter)
df["salario_min"] = df["salario_min"].str.replace(",", ".").astype(float)

#Converter n_filhos para inteiro (preencher NaN como 0)
df["n_filhos"] = df["n_filhos"].fillna(0).astype(int)

#Garantir que a idade é int
df["idade_anos"] = df["idade_anos"].astype(int)

#Tratar regiao_procedencia como categoria
df["regiao_procedencia"] = df["regiao_procedencia"].astype("category")

#Conferir resultado
df.info()
df.head()

#estatisticas gerais
df.describe(include="all")

#Média de salario por escolaridade
df.groupby("grau_instrucao")["salario_min"].mean()

#media de idade por estado civil
df.groupby("estado_civil")["idade_anos"].mean()

#quantidade de pessoas por cidade
df["regiao_procedencia"].value_counts()

#correlação
df.corr(numeric_only=True)

#Gráfico distribuição das idades
df["idade_anos"].plot(kind="hist", bins=10, figsize=(6,4))
plt.title("Distribuição das Idades")
plt.xlabel("Idade")
plt.show()

#Gráfico salario por escolaridade
df.groupby("grau_instrucao")["salario_min"].mean().plot(kind="bar")
plt.title("Salário médio por Escolaridade")
plt.ylabel("Salários mínimos")
plt.show()

#Gráfico de número de filhos
df["n_filhos"].value_counts().plot(kind="bar")
plt.title("Distribuição de Número de Filhos")
plt.xlabel("Filhos")
plt.ylabel("Quantidade")
plt.show()

#Gráfico de pessoas por cidade
df["regiao_procedencia"].value_counts().plot(kind="bar")
plt.title("Pessoas por Região de Procedência")
plt.show()

#Gráfico de idade X salário
plt.scatter(df["idade_anos"], df["salario_min"], alpha=0.7)
plt.title("Idade x Salário")
plt.xlabel("Idade")
plt.ylabel("Salários mínimos")
plt.show()

#análise avançada Regressão Linear - Prever Salário

X = df[["idade_anos", "n_filhos", "grau_instrucao", "estado_civil", "regiao_procedencia"]]
y = df["salario_min"]

#Categóricas → OneHotEncoding
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), ["grau_instrucao", "estado_civil", "regiao_procedencia"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))
#essa parte mostra quais varíaveis influenciam o salário
#quao bem o modelo prevê o salário
#se escolaridade e idade são relevantes ou não



#Clusterização (K-Means) - Grupos Sociais

df_cluster = df[["idade_anos", "n_filhos", "salario_min"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(df_cluster)

kmeans = KMeans(n_clusters=3, random_state=0)
df["cluster"] = kmeans.fit_predict(scaled)

print(df.groupby("cluster").mean(numeric_only=True))
#essa parte mostra grupo de jovens com baixa renda,
#adultos com filhos e renda intermediária,
#pessoas mais velhas com renda maior



#Feature Importance - O que realmente importa no salario

X = df[["idade_anos", "n_filhos", "grau_instrucao", "estado_civil", "regiao_procedencia"]]
y = df["salario_min"]

X_encoded = pd.get_dummies(X, drop_first=True)

model = RandomForestRegressor()
model.fit(X_encoded, y)

importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
print(importances.sort_values(ascending=False))
#essa parte mostra o que pesa mais no salário
#se a idade importa ou não
#se a escolaridade importa ou não
#se a cidade faz diferença ou não

#Correlação Avançada

corr = df[["idade_anos", "n_filhos", "salario_min"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
#essa parte mostra quanto a idade está ligada ao salário
#se mais filhos = menor salário
#relação entre irmãos e idade

X = df[["idade_anos", "n_filhos", "salario_min", "regiao_procedencia"]]
y = df["estado_civil"]

X_encoded = pd.get_dummies(X, drop_first=True)

model = LogisticRegression(max_iter=500)
model.fit(X_encoded, y)

print("Acurácia:", model.score(X_encoded, y))
#essa parte prevê o estado civil baseado nas características da pessoa

#Análise de outliers - pessoas "fora do padrão"

Q1 = df["salario_min"].quantile(0.25)
Q3 = df["salario_min"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["salario_min"] < Q1 - 1.5*IQR) | (df["salario_min"] > Q3 + 1.5*IQR)]
outliers
#essa parte mostra pessoas com renda extremamente alta
#pessoas com renda anormalmente baixa
