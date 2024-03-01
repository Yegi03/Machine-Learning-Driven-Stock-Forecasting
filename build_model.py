# build_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape=(60, 1)):
    """
    Builds an LSTM model tailored for time series forecasting, such as stock price prediction.
    
    Parameters:
    - input_shape: tuple, indicating the shape of the input data (timesteps, features).
      This is set to (60, 1) by default, suitable for looking at 60 days of stock prices to predict the next day.
    
    Returns:
    - A compiled TensorFlow Keras model ready for training.
    """
    model = Sequential([
        # First LSTM layer with return_sequences=True to feed into another LSTM layer
        LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape),
        # Dropout layer to reduce overfitting
        Dropout(rate=0.2),
        # Second LSTM layer, return_sequences=False as we do not need to return sequences
        LSTM(units=50, activation='relu'),
        # Another Dropout layer
        Dropout(rate=0.2),
        # Dense layer to output the predicted value
        Dense(units=1)
    ])
    
    # Compiling the model with mean squared error loss and the Adam optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

if __name__ == "__main__":
    # Ensuring reproducibility by setting a random seed
    tf.random.set_seed(42)
    
    # Building the model
    model = build_lstm_model()
    
    # Printing the model summary to verify the architecture
    model.summary()
