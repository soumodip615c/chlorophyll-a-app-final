# model_utils.py

# (your imports remain the same)

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

# Call during import so it's ready before request
run_prediction_and_plot()
