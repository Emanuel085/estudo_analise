#importar as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#configuraçao de estilo
sns.set(style="whitegrid")
#carregar o csv
df = pd.read_csv("titanic.csv", sep=",", engine="python")

print(df.head())
print(df.info())

#Estatisticas descritivas
print("\n Estatísticas Descritivas ")
print(df.describe(include="all"))

#Distribuição de sobreviventes
sobrevivencia = df["Survived"].value_counts()

print("\n Distribuição de Sobreviventes ")
print(sobrevivencia)

#Gráfico
plt.figure(figsize=(6,4))
sobrevivencia.plot(kind="bar")
plt.title("Sobrevivência no Titanic")
plt.xticks([0, 1], ["Morreu", "Sobreviveu"])
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

#Sobrevivência por sexo
surv_por_sexo = df.groupby("Sex")["Survived"].mean()

print("\n Taxa de sobrevivencia por sexo ")
print(surv_por_sexo)

#Gráfico
plt.figure(figsize=(6,4))
surv_por_sexo.plot(kind="bar")
plt.title("Sobrevivência por Sexo")
plt.ylabel("Taxa de sobrevivência")
plt.tight_layout()
plt.show()

surv_classe = df.groupby("Pclass")["Survived"].mean()

print("\n Taxa de sobrevivência por Classe ")
print(surv_classe)

#Gráfico
plt.figure(figsize=(6,4))
surv_classe.plot(kind="bar")
plt.title("Sobrevivência por Classe")
plt.xlabel("Classe")
plt.ylabel("Taxa")
plt.tight_layout()
plt.show()

#Idade média por sobrevivência
idade_por_surv = df.groupby("Survived")["Age"].mean()

print("\n Idade média por sobrevivência ")
print(idade_por_surv)

#Gráfico
plt.figure(figsize=(8,4))
df["Age"].hist(bins=20)
plt.title("Distribuição das Idades")
plt.xlabel("Idade")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

#Criar coluna pro tamanho da família
df["FamilySize"] = df["Siblings/Spouses Aboard"] + df["Parents/Children Aboard"] + 1

#Gráfico pra ver distribuição do tamanho da família
df["FamilySize"].value_counts().sort_index().plot(kind="bar", figsize=(7,4))
plt.title("Distribuição do Tamanho da Família")
plt.xlabel("Tamanho da Família")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

#Sobrevivencia por tamanho da família
df.groupby("FamilySize")["Survived"].mean().plot(kind="bar", figsize=(7,4))
plt.title("Taxa de sobrevivência por Tamanho da Família")
plt.xlabel("Tamanho da Família")
plt.ylabel("Probabilidade de sobrevivência")
plt.tight_layout()
plt.show()

#Sobrevivência por cada uma separadamente
df.groupby("Siblings/Spouses Aboard")["Survived"].mean()

df.groupby("Parents/Children Aboard")["Survived"].mean()

#Relação de idade e tarifa
plt.figure(figsize=(6,4))
plt.scatter(df["Age"], df["Fare"], alpha=0.5)
plt.title("Idade x Tarifa")
plt.xlabel("Idade")
plt.ylabel("Tarifa")
plt.tight_layout()
plt.show()

#Sobrevivência cruzada por sexo e classe
tabela = pd.crosstab(df["Sex"], df["Pclass"], values=df["Survived"], aggfunc="mean")

print("\n Sobrevivência por Sexo + Classe ")
print(tabela)