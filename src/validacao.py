import os
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers, regularizers, callbacks

BASE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(BASE, "..", "data", "preparados")
IMG = os.path.join(BASE, "..", "data", "images", "validacao")

X_treino = pd.read_csv(os.path.join(PREP, "X_treino.csv")).values
X_val = pd.read_csv(os.path.join(PREP, "X_val.csv")).values
y_treino = pd.read_csv(os.path.join(PREP, "y_treino.csv")).values.ravel() - 1
y_val = pd.read_csv(os.path.join(PREP, "y_val.csv")).values.ravel() - 1

# Analise das curvas existentes
print("=" * 50)
print("Análise das curvas de aprendizado (treinamento.py)")
print("=" * 50)

print("""
Observações:
  - Loss de treino e validação caem juntas = sem overfitting
  - Accuracy atinge 100% rapidamente (época 3) = convergência rápida
  - Curvas não divergem = modelo generaliza bem
  - Dados são 4 clusters bem separados = esperado convergir tão rápido

Conclusão: modelo saudável, sem overfitting nem underfitting.
""")

# Early Stopping
print("=" * 50)
print("Demonstração: Early Stopping")
print("=" * 50)

model_es = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(4, activation="softmax"),
])

model_es.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

hist_es = model_es.fit(
    X_treino, y_treino,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

epocas_es = len(hist_es.history["loss"])
print(f"Treino interrompido na época {epocas_es} (paciência=5)")
print(f"Loss final treino:   {hist_es.history['loss'][-1]:.6f}")
print(f"Loss final validação: {hist_es.history['val_loss'][-1]:.6f}")
print(f"Accuracy final validação: {hist_es.history['val_accuracy'][-1]:.4f}")

# L2 Regularization
print("\n" + "=" * 50)
print("Demonstração: L2 Regularization")
print("=" * 50)

model_l2 = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(32, activation="relu",
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(4, activation="softmax"),
])

model_l2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

hist_l2 = model_l2.fit(
    X_treino, y_treino,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=0,
)

print(f"Loss final treino (L2):   {hist_l2.history['loss'][-1]:.6f}")
print(f"Loss final validação (L2): {hist_l2.history['val_loss'][-1]:.6f}")
print(f"Accuracy final validação (L2): {hist_l2.history['val_accuracy'][-1]:.4f}")

# Graficos comparativos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Early Stopping - Loss
ax = axes[0, 0]
ax.plot(hist_es.history["loss"], label="Treino")
ax.plot(hist_es.history["val_loss"], label="Validação")
ax.axvline(x=epocas_es - 1, color="red", linestyle="--", alpha=0.5, label="Parada")
ax.set_xlabel("Épocas")
ax.set_ylabel("Loss")
ax.set_title("Early Stopping — Loss")
ax.legend()

# Early Stopping - Accuracy
ax = axes[0, 1]
ax.plot(hist_es.history["accuracy"], label="Treino")
ax.plot(hist_es.history["val_accuracy"], label="Validação")
ax.axvline(x=epocas_es - 1, color="red", linestyle="--", alpha=0.5, label="Parada")
ax.set_xlabel("Épocas")
ax.set_ylabel("Accuracy")
ax.set_title("Early Stopping — Accuracy")
ax.legend()

# L2 - Loss
ax = axes[1, 0]
ax.plot(hist_l2.history["loss"], label="Treino (L2)")
ax.plot(hist_l2.history["val_loss"], label="Validação (L2)")
ax.set_xlabel("Épocas")
ax.set_ylabel("Loss")
ax.set_title("L2 Regularization — Loss")
ax.legend()

# L2 - Accuracy
ax = axes[1, 1]
ax.plot(hist_l2.history["accuracy"], label="Treino (L2)")
ax.plot(hist_l2.history["val_accuracy"], label="Validação (L2)")
ax.set_xlabel("Épocas")
ax.set_ylabel("Accuracy")
ax.set_title("L2 Regularization — Accuracy")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG, "validacao.png"))
plt.close()
print(f"\nGráficos salvos em data/images/validacao/validacao.png")

print("\n" + "=" * 50)
print("Resumo da validação")
print("=" * 50)
print("""
1. Curvas de aprendizado: sem overfitting/underfitting
   - Treino e validação convergem juntos
   - Modelo atinge 100% accuracy rapidamente

2. Early Stopping: funcionalidade demonstrada
   - Com paciência=5, treino para assim que val_loss estabiliza
   - Útil quando não se sabe o número ideal de épocas

3. L2 Regularization: técnica demonstrada
   - Penaliza pesos grandes (kernel_regularizer)
   - Neste dataset simples não era necessária, mas é essencial
     em problemas reais com risco de overfitting
""")

model_es.save(os.path.join(BASE, "..", "data", "models", "modelo_validado.keras"))
print("Modelo com Early Stopping salvo em data/models/modelo_validado.keras")
