"""Entrena la red densa de Fashion-MNIST y guarda prendas.keras.

Es el mismo entrenamiento del cuaderno "introduccion keras.ipynb", sin depender
de Google Colab ni de Google Drive.

Uso:
    python entrenar.py            # 30 épocas (como en el cuaderno)
    python entrenar.py 10         # otro número de épocas
"""
import sys
from pathlib import Path

import tensorflow as tf

EPOCAS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SALIDA = Path(__file__).parent / "prendas.keras"

# 1. Datos: 60 000 imágenes de entrenamiento y 10 000 de prueba, 28 x 28 en grises.
#    Keras las descarga la primera vez (~30 MB) en ~/.keras/datasets/.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Normalización de píxeles a [0, 1] (la app hace lo mismo antes de predecir).
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Arquitectura: Flatten + 3 capas densas ReLU + salida softmax de 10 clases.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Entrenamiento.
model.fit(train_images, train_labels, epochs=EPOCAS)

# 5. Evaluación sobre el conjunto de prueba.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nExactitud en prueba: {test_acc:.4f}")

# 6. Guardado en el formato que carga app.py.
model.save(SALIDA)
print(f"Modelo guardado en {SALIDA}")
