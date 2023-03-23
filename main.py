import datetime
import os
import time
from keras.models import load_model
from train_model import train_lstm_model
from trade import execute_trading,load_preprocessed_data, get_latest_data, scale_data, make_prediction, trading_strategy, symbol_list, models
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

retrain_interval_days = 7
last_retrain_date = datetime.date.today()



while True:
    current_date = datetime.date.today()
    
    if (current_date - last_retrain_date).days >= retrain_interval_days:
        print("Retraining LSTM models...")
        n_steps_in = 60  # Use 60 time steps for input sequence
        n_steps_out = 1  # Predict 1 time step ahead
        n_features = 5   # Assume there are 5 features in the input data

                
        for symbol in symbol_list:
            data = load_preprocessed_data(symbol)
            train_lstm_model(data, f"{symbol[:-4]}_1d_lstm", n_steps_in, n_steps_out, n_features, 50, 0.2, 100, 32)

    # Execute the trading strategy
    print("Executing trading strategy...")
    execute_trading()

    # Sleep for 24 hours (86400 seconds) before checking again
    time.sleep(86400)



