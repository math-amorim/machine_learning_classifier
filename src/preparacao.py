import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(BASE, "..", "data", "preparados")

df = pd.read_csv(os.path.join(BASE, "..", "data", "dados_tratados.csv"))

# Escalonamento

X = df[["x", "y"]].values
y = df["classe"].values

print(f"Total de registros: {len(X)}")
print(f"Antes do escalonamento:")
print(f"  x: média={X[:,0].mean():.4f}, σ={X[:,0].std():.4f}")
print(f"  y: média={X[:,1].mean():.4f}, σ={X[:,1].std():.4f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nApós StandardScaler:")
print(f"  x: média={X_scaled[:,0].mean():.4f}, σ={X_scaled[:,0].std():.4f}")
print(f"  y: média={X_scaled[:,1].mean():.4f}, σ={X_scaled[:,1].std():.4f}")

# Separação 70/15/15

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Treino:   {len(X_treino)} ({len(X_treino)/len(X)*100:.1f}%)")
print(f"Validação: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Teste:    {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

for nome, X_conj, y_conj in [
    ("Treino", X_treino, y_treino),
    ("Validação", X_val, y_val),
    ("Teste", X_test, y_test),
]:
    contagem = pd.Series(y_conj).value_counts().sort_index()
    classes = dict(zip(contagem.index.astype(int), contagem.values))
    print(f"  {nome}: {len(X_conj)} registros, "
          f"média x={X_conj[:,0].mean():.4f}, σ x={X_conj[:,0].std():.4f} | "
          f"classes={classes}")

os.makedirs(SAIDA, exist_ok=True)

pd.DataFrame(X_treino, columns=["x", "y"]).to_csv(
    os.path.join(SAIDA, "X_treino.csv"), index=False)
pd.DataFrame(X_val, columns=["x", "y"]).to_csv(
    os.path.join(SAIDA, "X_val.csv"), index=False)
pd.DataFrame(X_test, columns=["x", "y"]).to_csv(
    os.path.join(SAIDA, "X_test.csv"), index=False)

pd.DataFrame(y_treino, columns=["classe"]).to_csv(
    os.path.join(SAIDA, "y_treino.csv"), index=False)
pd.DataFrame(y_val, columns=["classe"]).to_csv(
    os.path.join(SAIDA, "y_val.csv"), index=False)
pd.DataFrame(y_test, columns=["classe"]).to_csv(
    os.path.join(SAIDA, "y_test.csv"), index=False)

print(f"\nArquivos salvos em {SAIDA}/")
print("X_treino.csv  X_val.csv  X_test.csv  y_treino.csv  y_val.csv  y_test.csv")
