import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_preprocessed_data(filename):
    return pd.read_csv(f"data/preprocessed_{filename}")

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences) - n_steps_in - n_steps_out + 1):
        X.append(sequences[i : i + n_steps_in, :])
        y.append(sequences[i + n_steps_in : i + n_steps_in + n_steps_out, -1])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

symbol_list = ['BTCUSDT_1d.csv', 'ETHUSDT_1d.csv', 'BNBUSDT_1d.csv', 'XRPUSDT_1d.csv']

# Define the number of input and output steps for the LSTM model
n_steps_in, n_steps_out = 60, 1

for symbol in symbol_list:
    # Load preprocessed data
    data = load_preprocessed_data(symbol)

    # Scale and normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 1:].values)

    # Prepare input and output sequences
    X, y = split_sequences(scaled_data, n_steps_in, n_steps_out)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the LSTM model
    input_shape = (n_steps_in, X.shape[2])
    model = create_lstm_model(input_shape)
    train_lstm_model(model, X_train, y_train)

    # Save the trained model
    model.save(f"models/{symbol.split('.')[0]}_lstm.h5")
    print(f"Trained and saved LSTM model for {symbol}")
