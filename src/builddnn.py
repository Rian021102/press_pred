import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_dnn(X_train_scaled, y_train, X_test_scaled, y_test):
    # Ensure the input is in the correct shape
    input_shape = [X_train_scaled.shape[1]]  # Use list to ensure correct input shape
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        batch_size=32,
        epochs=100,
        verbose=1
    )

    return model, history

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mae(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mean_absolute_error'], label='train')
    plt.plot(history.history['val_mean_absolute_error'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pred(y_val, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100], 'r--')  # Red dashed line for perfect prediction
    plt.show()
