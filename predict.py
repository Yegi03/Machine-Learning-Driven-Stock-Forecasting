# predict.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from preprocess_data import load_data, create_sequences

def inverse_transform(scaler, data, column_index=0):
    """
    Inverse transforms the scaled data to its original scale.
    """
    dummy = np.zeros(shape=(len(data), scaler.feature_range[1]))
    dummy[:,column_index] = data[:,0]
    return scaler.inverse_transform(dummy)[:,column_index]

if __name__ == "__main__":
    # Load the saved model
    model = load_model('best_model.h5')
    
    # Load and preprocess the data
    file_path = 'AAPL_stock_data.csv'
    data = load_data(file_path)
    
    # Assuming 'Close' is the column index 0 after loading
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])
    
    # Create sequences from the scaled data
    sequence_length = 60
    X_test, y_test = create_sequences(data_scaled, sequence_length)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values to original scale
    predictions_inverse = inverse_transform(scaler, predictions)
    y_test_inverse = inverse_transform(scaler, y_test)
    
    # Example: Compare the first 5 predicted vs actual values
    for i in range(5):
        print(f"Predicted: {predictions_inverse[i]}, Actual: {y_test_inverse[i]}")
