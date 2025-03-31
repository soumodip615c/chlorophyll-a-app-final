import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def run_prediction_and_plot():
    model = load_model('models/pretrained_model.h5')

    # Correct input shape: (1, 10, 10, 10, 1)
    input_data = np.random.rand(1, 10, 10, 10, 1)

    # Run prediction
    prediction = model.predict(input_data)

    # Plot the first frame of the prediction
    plt.figure(figsize=(6, 5))
    plt.imshow(prediction[0, 0, :, :, 0], cmap='viridis')
    plt.title("Predicted Chlorophyll-a Frame")
    plt.colorbar()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("static/prediction_plot.png")
    plt.close()

