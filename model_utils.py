# model_utils.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization

# ✅ Define helper functions first
def generate_demo_data(shape=(10, 64, 64)):
    data = np.random.rand(*shape)
    data[data < 0.2] = np.nan
    return data

def load_and_preprocess(data):
    mask = np.isnan(data)
    data[mask] = np.nanmean(data)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    return data_scaled, scaler

def create_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)[..., np.newaxis]
    return X, y

def build_model(input_shape):
    model = Sequential([
        ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(1, (1, 1), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ✅ Now use the above functions
def run_prediction_and_plot():
    raw_data = generate_demo_data()
    data_scaled, scaler = load_and_preprocess(raw_data)
    X, y = create_sequences(data_scaled)
    model = build_model((X.shape[1], X.shape[2], X.shape[3], 1))
    model.fit(X, y, epochs=1, batch_size=2, validation_split=0.2, verbose=0)
    predicted = model.predict(X)
    predicted_rescaled = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(predicted.shape)

    # Save the plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(y[0, :, :, 0], cmap='viridis')
    plt.title("Original Chl-a")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_rescaled[0, :, :, 0], cmap='viridis')
    plt.title("Predicted Chl-a (AI-Filled)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("static/plot.png")
    plt.close()
