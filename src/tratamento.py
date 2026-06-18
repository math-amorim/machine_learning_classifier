import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE, "..", "data", "dados.csv"),
    sep="\t",
    decimal=",",
    comment="#",
    header=None,
    names=["x", "y", "classe"]
)

linhas_antes = len(df)

print(f"Registros antes da remoção: {linhas_antes}")

print("\nValores nulos por coluna:")
print(df.isnull().sum())

total_nulos = df.isnull().sum().sum()
print(f"\nTotal de valores nulos: {total_nulos} ({total_nulos / linhas_antes * 100:.2f}% dos dados)")

print("\nLinhas com NaN:")
linhas_nan = df[df.isnull().any(axis=1)]
print(linhas_nan)

df_tratado = df.dropna()
linhas_depois = len(df_tratado)
removidas = linhas_antes - linhas_depois

print(f"\nRegistros removidos: {removidas}")
print(f"Registros após remoção: {linhas_depois}")
print(f"Perda: {removidas / linhas_antes * 100:.2f}% dos dados")

print("\nVerificação pós remoção:")
print(df_tratado.isnull().sum())

saida = os.path.join(BASE, "..", "data", "dados_tratados.csv")
print(f"\nSalvando dados tratados em {saida} ({linhas_depois} registros)")
df_tratado.to_csv(saida, index=False)

print("Concluído.")
