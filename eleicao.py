#IMPORTAR BIBLIOTECAS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Configuração de estilo
sns.set(style="whitegrid")

#Carregar o csv

df = pd.read_csv("eleicoes.csv", sep=",", engine="python")
print(df.columns.to_list())

#remover espaço nas colunas

df.columns = [c.strip() for c in df.columns]

#Converter colunas numéricas

df["Money (R$ Reais)"] = df["Money (R$ Reais)"].astype(float)
df["Votes"] = df["Votes"].astype(int)

#remover espaços dos valores de estado

df["State"] = df["State"].str.strip()

print(df.head())

#estatisticas basicas

print("\nEstatísticas gerais")
print(df.describe())

#Media por estado
print("\nMédia de votos por estado")
print(df.groupby("State")["Votes"].mean())

print("\n Média de gastos por estado")
print(df.groupby("State")["Money (R$ Reais)"].mean())

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Money (R$ Reais)", y="Votes", alpha=0.6)
plt.title("Correlação entre gastos e votos")
plt.show()

#Cálculo da correlação
correl = df["Money (R$ Reais)"].corr(df["Votes"])
print(f"\nCorrelação gasto x votos: {correl:.4f}")

#Ranking de candidatos

print("\n Top 10 mais votados")
print(df.nlargest(10, "Votes"))

print("\n Top 10 que mais gastaram")
print(df.nlargest(10, "Money (R$ Reais)"))

#Eficiencia da campanha
df["Eficiencia (votos por real)"] = df["Votes"] / df["Money (R$ Reais)"]

print("\n Top 10 campanhas mais eficientes")
print(df.nlargest(10, "Eficiencia (votos por real)"))

#Comparações por estado

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="State", y="Money (R$ Reais)")
plt.title("Distribuição dos gastos por estado")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="State", y="Votes")
plt.title("Distribuição dos votos por estado")
plt.xticks(rotation=90)
plt.show()

#Distribuições e histogramas

plt.figure(figsize=(7,4))
sns.histplot(df["Money (R$ Reais)"], bins=30, kde=True)
plt.title("Distribuição dos gastos")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df["Votes"], bins=30, kde=True)
plt.title("Distribuição dos votos")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df["Eficiencia (votos por real)"], bins=30, kde=True)
plt.title("Distribuição da eficiência")
plt.show()

#Custo por voto

df["Custo por voto"] = df["Money (R$ Reais)"] / df["Votes"]

print("\n Campanhas com menor custo por voto")
print(df.nsmallest(10, "Custo por voto"))

plt.figure(figsize=(7,4))
sns.histplot(df["Custo por voto"], bins=30, kde=True)
plt.title("Custo por voto - distribuição")
plt.show()

#Regressão simples
plt.figure(figsize=(8,5))
sns.regplot(data=df, x="Money (R$ Reais)", y="Votes", scatter_kws={"alpha":0.5})
plt.title("Regressão: votos em função dos gastos")
plt.show()