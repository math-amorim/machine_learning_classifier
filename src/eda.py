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

print("INFORMAÇÕES GERAIS")
print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")

print("\nTIPOS DAS COLUNAS")
print(df.dtypes)

print("\nVALORES NULOS")
print(df.isnull().sum())

print("\nDUPLICADOS")
print(df.duplicated().sum())

print("\nDISTRIBUIÇÃO DAS CLASSES")
print(df["classe"].value_counts().sort_index())

print("\nESTATÍSTICAS DESCRITIVAS")
print(df.describe())


print("\nGERANDO GRÁFICOS...")
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
plt.savefig("../data/images/distribuica_classes.png")
plt.close()

plt.figure(figsize=(8,8))
df["x"].hist(bins=30)
plt.xlabel("X")
plt.ylabel("Frequência")
plt.title("Histograma X")
plt.savefig("../data/images/histograma_x.png")
plt.close()

plt.figure(figsize=(8,8))
df["y"].hist(bins=30)
plt.xlabel("Y")
plt.ylabel("Frequência")
plt.title("Histograma Y") 
plt.savefig("../data/images/histograma_y.png")
plt.close()

plt.figure(figsize=(8,8))
df[["x", "y"]].boxplot()
plt.title("Boxplot")
plt.savefig("../data/images/boxplot.png")
plt.close()

plt.figure(figsize=(8,8))
corr = df[["x","y"]].corr()
plt.imshow(corr)

plt.xticks([0,1], ["x","y"])
plt.yticks([0,1], ["x","y"])

plt.colorbar()
plt.savefig("../data/images/correlacao.png")
plt.close()

print("\nGRÁFICOS GERADOS EM data/images")
