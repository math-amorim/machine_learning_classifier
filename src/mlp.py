import os
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Arquitetura da MLP")
print(f"Camadas: {len(model.layers)}")
print(f"Parâmetros treináveis: {model.count_params()}")
print()

model.summary()

saida = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "models", "modelo.keras",
)
model.save(saida)
print(f"\nModelo salvo em {saida}")