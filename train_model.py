import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt


X_train = np.load("models/X_train.npy")
y_train = np.load("models/y_train.npy")
X_val   = np.load("models/X_val.npy")
y_val   = np.load("models/y_val.npy")
X_test  = np.load("models/X_test.npy")
y_test  = np.load("models/y_test.npy")
classes = np.load("models/classes.npy", allow_pickle=True)

NUM_CLASSES = len(classes)
N_FRAMES    = X_train.shape[1]
N_FEATURES  = X_train.shape[2]

print(f"Trening: {X_train.shape}, Test: {X_test.shape}")
print(f"Broj klasa: {NUM_CLASSES}")
print(f"Broj feature-a po frejmu: {N_FEATURES}")

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val, NUM_CLASSES)
y_test_cat  = to_categorical(y_test, NUM_CLASSES)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(N_FRAMES, N_FEATURES)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True)
]

print("Model training started (body normalization + curl features)...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=60,
    batch_size=32,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test set accuracy of the model: {accuracy*100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during training (V3)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print("Training completed! Model saved as models/best_model.keras")