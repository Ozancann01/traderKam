import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import os
from simulate import simulate_trades
from strategies import momentum,swing_trading,technical_analysis,trend_following,mean_reversion,breakout

from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator
import datetime
from train_model import train_lstm_model





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
    
    




def trading_strategy(strategy_name, prediction, historical_data, period=14):
  
    if strategy_name == 'momentum':
        signal = momentum(historical_data, period=14)
    elif strategy_name == 'swing_trading':
        signal = swing_trading(historical_data, period)
    elif strategy_name == 'technical_analysis':
        signal = technical_analysis(historical_data, period)
    elif strategy_name == 'trend_following':
        signal = trend_following(historical_data, period)
    elif strategy_name == 'mean_reversion':
        signal = mean_reversion(historical_data, period)
    elif strategy_name == 'breakout':
        signal = breakout(historical_data, period)
    else:
        raise ValueError("Invalid strategy_name")

    data = historical_data.copy()
    data["Signal"] = signal

    return data


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


def evaluate_strategy(strategy_name,prediction, historical_data, period=14):
    strategy_data = trading_strategy(strategy_name,prediction, historical_data, period)
    profit_loss = simulate_trades(strategy_data)
    return profit_loss



def execute_trading():
    global last_retrain_date
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
        
            # Call store_trade_info function to save trade information
        trade_data = {'symbol': model_name, 'date': datetime.date.today(), 'prediction': prediction}
        store_trade_info(model_name, trade_data)
        
     # Check if it's time to retrain the models
        current_date = datetime.date.today()
        days_since_last_retrain = (current_date - last_retrain_date).days
        if days_since_last_retrain >= retrain_interval_days:
            print(f"Retraining {model_name} model...")
            train_lstm_model(model_name, data)
            models[model_name] = load_model(f"models/{model_name}_lstm.h5")
            last_retrain_date = current_date


def calculate_position_size(balance, entry_price, stop_loss_price, risk_percentage):
    risk_amount = balance * risk_percentage
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size





def calculate_position_size(balance, entry_price, stop_loss_price, risk_percentage):
    risk_amount = balance * risk_percentage
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size

def execute_trading():
    risk_percentage = 0.01  # Set your desired risk percentage, e.g., 1%
    stop_loss_percentage = 0.03  # Set your desired stop loss percentage, e.g., 3%
    balance = 1000  # You need to fetch your actual account balance using your trading platform's API

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
        
        # Calculate entry price, stop loss price, and position size
        entry_price = prediction
        stop_loss_price = entry_price * (1 - stop_loss_percentage)
        position_size = calculate_position_size(balance, entry_price, stop_loss_price, risk_percentage)
        
        # TODO: Implement the logic to place orders based on the signal and position size
        # You need to define your trading platform's API to place orders
        # For example, if you are using Binance:
        # if strategy_data.iloc[-1]['Signal'] == 1:  # Buy signal
        #     place_order('buy', symbol, position_size)
        # elif strategy_data.iloc[-1]['Signal'] == -1:  # Sell signal
        #     place_order('sell', symbol, position_size)
