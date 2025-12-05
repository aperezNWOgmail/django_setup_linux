# apollo_predictor/model_loader.py
import tensorflow as tf
import numpy as np
import threading

# Global variable to hold the model
_model = None
_model_lock = threading.Lock()  # Thread safety for model loading


def load_model():
    global _model
    with _model_lock:
        if _model is None:
            print("Loading TensorFlow model for Apollo prediction...")
            # --- Define and Train the Model (Same as before) ---
            historical_data = np.array([
                [8.0,  147.0],  # Apollo 8
                [10.0, 193.0],  # Apollo 10
                [11.0, 195.0],  # Apollo 11
                [12.0, 244.0],  # Apollo 12
                [13.0, 142.0],  # Apollo 13
                [14.0, 217.0],  # Apollo 14
                [15.0, 295.0],  # Apollo 15
                [16.0, 265.0],  # Apollo 16
                [17.0, 301.0]  # Apollo 17
            ])

            X = historical_data[:, 0].reshape(-1, 1)
            y = historical_data[:, 1]

            _model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1, input_shape=[1])
            ])

            _model.compile(optimizer=tf.keras.optimizers.SGD(
                learning_rate=0.01), loss='mean_squared_error')
            _model.fit(X, y, epochs=1000, verbose=0)
            print("Model loaded successfully.")

    return _model


def predict_time(mission_number):
    model = load_model()
    # Ensure input is in the correct shape for prediction
    input_array = np.array([[mission_number]], dtype=np.float32)
    prediction = model.predict(input_array, verbose=0)
    # Extract the scalar value from the prediction array
    return float(prediction[0][0])

# --- Example Usage (Optional) ---
# if __name__ == "__main__":
#     time = predict_time(18.0)
#     print(f"Predicted time for Apollo 18: {time} hours")
