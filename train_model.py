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




def trading_strategy(predictions, historical_data, window=14):
    """
    This function takes the predictions from the predict_direction function, historical data of the asset, and
    a window for computing the moving average. It then creates a trading strategy based on the predicted
    price direction and moving averages.

    Args:
    predictions (list): A list of predicted price directions.
    historical_data (pd.DataFrame): A DataFrame containing historical data for the asset.
    window (int): The window size for computing the moving average.

    Returns:
    pd.DataFrame: A DataFrame containing buy and sell signals.
    """
    # Calculate the moving average
    historical_data['MovingAverage'] = historical_data['Close'].rolling(window=window).mean()

    # Create a new DataFrame for the trading strategy
    strategy = pd.DataFrame(index=historical_data.index)
    strategy['Close'] = historical_data['Close']
    strategy['MovingAverage'] = historical_data['MovingAverage']
    strategy['Prediction'] = 0
    strategy['Prediction'].iloc[window:] = predictions

    # Generate buy and sell signals
    strategy['Signal'] = 0
    strategy.loc[strategy['Prediction'] == 1, 'Signal'] = 1
    strategy.loc[strategy['Prediction'] == -1, 'Signal'] = -1

    # Remove any rows that don't have a complete moving average
    strategy.dropna(inplace=True)

    return strategy




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
