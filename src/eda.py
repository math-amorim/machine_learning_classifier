import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "../data/dados.csv",
    sep="\t",
    decimal=",",
    comment="#",
    header=None,
    names=["x", "y", "classe"]
).dropna()

print("Informações Gerais")
print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")

print("\nTipos das colunas")
print(df.dtypes)

print("\Valores nulos")
print(df.isnull().sum())

print("\Duplicados")
print(df.duplicated().sum())

print("\nDistribuição das classes")
print(df["classe"].value_counts().sort_index())

print("\nEstatísticas Descritivas")
print(df.describe())


print("\nGerando gráficos...")
plt.figure(figsize=(8,8))
for classe in sorted(df["classe"].unique()):
    subset = df[df["classe"] == classe] 

    plt.scatter(
            subset["x"],
            subset["y"],
            label=f"Classe {int(classe)}",
            alpha=0.7
    )
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Distribuição das Classes")
plt.legend()
plt.savefig("../data/images/eda/distribuica_classes.png")
plt.close()

plt.figure(figsize=(8,8))
df["x"].hist(bins=30)
plt.xlabel("X")
plt.ylabel("Frequência")
plt.title("Histograma X")
plt.savefig("../data/images/eda/histograma_x.png")
plt.close()

plt.figure(figsize=(8,8))
df["y"].hist(bins=30)
plt.xlabel("Y")
plt.ylabel("Frequência")
plt.title("Histograma Y") 
plt.savefig("../data/images/eda/histograma_y.png")
plt.close()

plt.figure(figsize=(8,8))
df[["x", "y"]].boxplot()
plt.title("Boxplot")
plt.savefig("../data/images/eda/boxplot.png")
plt.close()

plt.figure(figsize=(8,8))
corr = df[["x","y"]].corr()
plt.imshow(corr)

plt.xticks([0,1], ["x","y"])
plt.yticks([0,1], ["x","y"])

plt.colorbar()
plt.savefig("../data/images/eda/correlacao.png")
plt.close()

print("\nGráficos gerados em data/images/eda")
