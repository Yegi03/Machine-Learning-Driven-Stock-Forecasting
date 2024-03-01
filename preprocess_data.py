import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

def create_sequences(data, sequence_length):
    """
    Creates sequences from the data.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def preprocess_data(stock_data, sequence_length=60):
    """
    Preprocesses the stock data for modeling.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_data('AAPL_stock_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print("Data preprocessed.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
