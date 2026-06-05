import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt


X_train = np.load("models/X_train.npy")
y_train = np.load("models/y_train.npy")
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")
classes = np.load("models/classes.npy", allow_pickle=True)



NUM_CLASSES = len(classes)
print(f"Trening: {X_train.shape}, Test: {X_test.shape}")
print(f"Broj klasa: {NUM_CLASSES}")


y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape = (40, 258)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True)
]

print("Pocetak treniranja")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Tacnost modela na test setu {accuracy*100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],    label='Trening')
plt.plot(history.history['val_accuracy'], label='Validacija')
plt.title('Tacnost tokom treninga')
plt.xlabel('Epoha')
plt.ylabel('Tacnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],    label='Trening')
plt.plot(history.history['val_loss'], label='Validacija')
plt.title('Greska tokom treninga')
plt.xlabel('Epoha')
plt.ylabel('Greska')
plt.legend()

plt.tight_layout()
plt.savefig('trening_rezultati.png')
plt.show()

print("Gotovo! Model sacuvan u models/best_model.keras")