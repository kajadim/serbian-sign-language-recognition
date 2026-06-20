import numpy as np
import os


os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping



def build_model(hp):
    model = Sequential()
    
    # Prvi LSTM sloj 
    hp_lstm_1 = hp.Int('lstm_1_units', min_value=64, max_value=256, step=64)
    model.add(LSTM(units=hp_lstm_1, return_sequences=True, input_shape=(40, 258)))
    
    # Prvi Dropout 
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))
    
    # Drugi LSTM sloj 
    hp_lstm_2 = hp.Int('lstm_2_units', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_lstm_2, return_sequences=False))
    
    # Drugi Dropout
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_2))
    
    # Dense sloj 
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    model.add(Dense(units=hp_dense_units, activation='relu'))
    
    # Izlazni sloj 
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Biranje brzine učenja (Learning Rate)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("Učitavanje podataka za tjuning...")
    X_train = np.load("models/X_train.npy")
    y_train = np.load("models/y_train.npy")
    X_val = np.load("models/X_val.npy")
    y_val = np.load("models/y_val.npy")
    classes = np.load("models/classes.npy", allow_pickle=True)
    
    NUM_CLASSES = len(classes)
    
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='tuner_results',
        project_name='ssl_hyperparameters'
    )
    
    # EarlyStopping unutar tunera
    stop_early = EarlyStopping(monitor='val_accuracy', patience=3)
    
    print("\nZapočinje automatska pretraga najboljeg modela...")
    tuner.search(
        X_train, y_train_cat,
        epochs=20,
        validation_data=(X_val, y_val_cat),
        callbacks=[stop_early]
    )
    
    # Izvlačenje i ispis najboljih rezultata
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    best_model = tuner.hypermodel.build(best_hps)
    
    os.makedirs("models", exist_ok=True)
    best_model.save("models/best_params.keras")

    print("\n" + "="*40)
    print("PRETRAGA ZAVRŠENA! NAJBOLJI PARAMETRI SU:")
    print(f"-> Prvi LSTM sloj: {best_hps.get('lstm_1_units')} neurona")
    print(f"-> Prvi Dropout:    {best_hps.get('dropout_1'):.1f}")
    print(f"-> Drugi LSTM sloj: {best_hps.get('lstm_2_units')} neurona")
    print(f"-> Drugi Dropout:   {best_hps.get('dropout_2'):.1f}")
    print(f"-> Dense sloj:      {best_hps.get('dense_units')} neurona")
    print(f"-> Learning Rate:   {best_hps.get('learning_rate')}")
    print("="*40)