# train_model.py
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from build_model import build_lstm_model
from preprocess_data import load_data, preprocess_data

def train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    """
    Trains the LSTM model and evaluates its performance on the test set.
    """
    # Adjust the input shape parameter according to your data preprocessing
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Callbacks for early stopping and saving the model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    ]
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.2, callbacks=callbacks, verbose=1)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    
    return model, history

if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'AAPL_stock_data.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model and evaluate it
    model, history = train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=64)
    
    print("Model training completed.")
