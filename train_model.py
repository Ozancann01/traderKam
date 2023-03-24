import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)


def load_preprocessed_data(filename):
    return pd.read_csv(f"data/preprocessed_{filename}")

def get_latest_data(data, n_steps_in):
    return data.iloc[-n_steps_in:, 1:].values

def scale_data(data, scaler):
    return scaler.transform(data)

def make_prediction(model, data):
    data = np.expand_dims(data, axis=0)
    return model.predict(data)[0, 0]

symbol_list = ['BTCUSDT_1d.csv', 'ETHUSDT_1d.csv', 'BNBUSDT_1d.csv', 'XRPUSDT_1d.csv']
n_steps_in = 60

# Load the trained LSTM models
models = {}
for symbol in symbol_list:
    model_name = symbol.split('.')[0]
    models[model_name] = load_model(f"models/{model_name}_lstm.h5")


def train_lstm_model(data, symbol, n_steps_in, n_steps_out, n_features, lstm_units, dropout_rate, epochs, batch_size):
    # The existing implementation of your train_lstm_model function

    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data.iloc[i:i+n_steps_in, 1:].values)
        y.append(data.iloc[i+n_steps_in:i+n_steps_in+n_steps_out, 3].values)
    X, y = np.array(X), np.array(y)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_steps_in, data.shape[1]-1)))
    model.add(Dense(units=n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Save the trained model
    model.save(f"models/{symbol}_lstm.h5")

    return model


# Make predictions for each cryptocurrency pair
for symbol in symbol_list:
    data = load_preprocessed_data(symbol)
    latest_data = get_latest_data(data, n_steps_in)
    
    # Scale the latest data
    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:, 1:])
    scaled_data = scale_data(latest_data, scaler)
    
    # Make a prediction using the LSTM model
    model_name = symbol.split('.')[0]
    prediction = make_prediction(models[model_name], scaled_data)
    
    print(f"Prediction for {symbol}: {prediction}")

# TODO: Implement the trading_strategy function to make trading decisions based on the predictions
