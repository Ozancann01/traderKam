import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import os


print("TensorFlow version:", tf.__version__)


import datetime
import os

retrain_interval_days = 7
last_retrain_date = datetime.date.today()


def load_preprocessed_data(filename):
    return pd.read_csv(f"data/preprocessed_{filename}")

def get_latest_data(data, n_steps_in):
    return data.iloc[-n_steps_in:, 1:].values

def scale_data(data, scaler):
    return scaler.transform(data)

def make_prediction(model, data):
    data = np.expand_dims(data, axis=0)
    return model.predict(data)[0, 0]

def store_trade_info(symbol, trade_data):
    trade_history_file = f"data/trade_history_{symbol}.csv"
    
    if os.path.exists(trade_history_file):
        trade_history = pd.read_csv(trade_history_file)
        trade_history = trade_history.append(trade_data, ignore_index=True)
    else:
        trade_history = pd.DataFrame([trade_data])
    
    trade_history.to_csv(trade_history_file, index=False)


symbol_list = ['BTCUSDT_1d.csv', 'ETHUSDT_1d.csv', 'BNBUSDT_1d.csv', 'XRPUSDT_1d.csv']
n_steps_in = 60

# Load the trained LSTM models
models = {}
for symbol in symbol_list:
    model_name = symbol.split('.')[0]
    models[model_name] = load_model(f"models/{model_name}_lstm.h5")


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

models_ensemble = {
    'BTCUSDT': [
        load_model("models/BTCUSDT_1d_lstm.h5"),
    ],
    'ETHUSDT': [
        load_model("models/ETHUSDT_1d_lstm.h5"),
    ],
    'BNBUSDT': [
        load_model("models/BNBUSDT_1d_lstm.h5"),
    ],
    'XRPUSDT': [
        load_model("models/XRPUSDT_1d_lstm.h5"),
    ],
}



def execute_trading():
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


