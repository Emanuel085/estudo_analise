#Importar as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Configuração de estilo
sns.set(style="whitegrid")
#Carregar o csv
df = pd.read_csv("nycflights.csv", sep=",", engine="python")

print(df.head())
print(df.info())

#Remover espaço das colunas
df.columns = [c.strip() for c in df.columns]

#Estatisticas básicas
print("\n Estatísticas Básicas")
print(df.describe())

#Atraso média por companhia aérea (carrier)
atraso_companhia = df.groupby("carrier")["arr_delay"].mean().sort_values()
print("\n Atraso Médio por companhia aérea ")
print(atraso_companhia)

#Gráfico
plt.figure(figsize=(10,4))
atraso_companhia.plot(kind="bar")
plt.title("Atraso médio por Companhia Aérea")
plt.xlabel("Companhia")
plt.ylabel("Atraso médio (Arr Delay)")
plt.tight_layout()
plt.show()

#Atraso médio por aeroporto de origem
atraso_origem = df.groupby("origin")["dep_delay"].mean().sort_values()
print("\n Atraso médio por aeroporto ")
print(atraso_origem)

#Gráfico
plt.figure(figsize=(6,4))
atraso_origem.plot(kind="bar")
plt.title("Atraso médio por Aeroporto (Dep Delay)")
plt.xlabel("Aeroporto")
plt.ylabel("Atraso médio")
plt.tight_layout()
plt.show()

#Atraso médio por mês
atraso_mes = df.groupby("month")["arr_delay"].mean()
print("\n Atraso médio por mês ")
print(atraso_mes)

#Gráfico
plt.figure(figsize=(10,4))
atraso_mes.plot(kind="bar")
plt.title("Atraso médio por Mês")
plt.xlabel("Mês")
plt.ylabel("Atraso Médio")
plt.grid(True)
plt.tight_layout()
plt.show()

#Top 10 piores atrasos
top10 = df.nlargest(10, "arr_delay")[["carrier", "origin", "arr_delay",]]
print("\n Top 10 Maiores Atrasos ")
print(top10)

#Quantidade de voos por destino
voos_dest = df["dest"].value_counts()
print("\n Voos por destino ")
print(voos_dest)

#Gráfico
plt.figure(figsize=(14,5))
voos_dest.head(20).plot(kind="bar")
plt.title("20 Destinos com mais voos")
plt.xlabel("Destino")
plt.ylabel("Quantidade de voos")
plt.tight_layout()
plt.show()

#Correlação entre atraso de saída e de chegada
corr = df["dep_delay"].corr(df["arr_delay"])
print("\n Correlação dep_delay x arr_delay =", corr)

#Gráfico de dispersão
plt.figure(figsize=(6,4))
plt.scatter(df["dep_delay"], df["arr_delay"], alpha=0.3)
plt.title("Correlação entre Atraso de Saída e Chegada")
plt.xlabel("Atraso de Saída (dep_delay)")
plt.ylabel("Atraso de Chegada (arr_delay)")
plt.tight_layout()
plt.show()

#Relação entre distância e atraso
dist_atraso = df.groupby("distance")["arr_delay"].mean()
print("\n Atraso médio por distância ")
print(dist_atraso.head())

#Gráfico
plt.figure(figsize=(10,4))
plt.scatter(df["distance"], df["arr_delay"], alpha=0.2)
plt.title("Distância do voo x Atraso")
plt.xlabel("Distância")
plt.ylabel("Atraso (arr_delay)")
plt.tight_layout()
plt.show()