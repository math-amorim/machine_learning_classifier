import os
import pandas as pd
import matplotlib.pyplot as plt
import keras

BASE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(BASE, "..", "data", "preparados")
MODELO = os.path.join(BASE, "..", "data", "models", "modelo.keras")

X_treino = pd.read_csv(os.path.join(PREP, "X_treino.csv")).values
X_val = pd.read_csv(os.path.join(PREP, "X_val.csv")).values
y_treino = pd.read_csv(os.path.join(PREP, "y_treino.csv")).values.ravel()
y_val = pd.read_csv(os.path.join(PREP, "y_val.csv")).values.ravel()

y_treino = y_treino - 1
y_val = y_val - 1

print(f"Treino:   {len(X_treino)} registros, classes {set(y_treino)}")
print(f"Validação: {len(X_val)} registros, classes {set(y_val)}")

model = keras.models.load_model(MODELO)

print("\nHiperparâmetros:")
print("  Learning rate: 0.001 (Adam default)")
print("  Batch size: 32")
print("  Épocas: 100")

history = model.fit(
    X_treino, y_treino,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
)

epocas = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epocas, history.history["loss"], label="Treino")
plt.plot(epocas, history.history["val_loss"], label="Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Curva de Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epocas, history.history["accuracy"], label="Treino")
plt.plot(epocas, history.history["val_accuracy"], label="Validação")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.title("Curva de Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE, "..", "data", "images", "treinamento", "curvas_treinamento.png"))
plt.close()
print("\nGráfico salvo em data/images/treinamento/curvas_treinamento.png")

model.save(os.path.join(BASE, "..", "data", "models", "modelo_treinado.keras"))
print("Modelo treinado salvo em data/models/modelo_treinado.keras")
