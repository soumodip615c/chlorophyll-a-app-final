import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam

# Generate dummy training data
X_dummy = np.random.rand(5, 10, 10, 10, 1)
y_dummy = np.random.rand(5, 10, 10, 10, 1)

# Define a simple ConvLSTM model
model = Sequential([
    ConvLSTM2D(filters=8, kernel_size=(3, 3), input_shape=(10, 10, 10, 1),
               padding='same', return_sequences=True),
    BatchNormalization(),
    Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')
])

# ✅ Use full loss function name here
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train briefly
model.fit(X_dummy, y_dummy, epochs=1)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the model
model.save("models/pretrained_model.h5")

print("✅ Dummy ConvLSTM model saved at: models/pretrained_model.h5")
