import datetime
import os
import time
from simulate import simulate_trades
from sklearn.preprocessing import MinMaxScaler

from trade import  evaluate_strategy


from keras.models import load_model
from train_model import train_lstm_model
from trade import execute_trading,load_preprocessed_data, get_latest_data, scale_data, make_prediction, trading_strategy, symbol_list, models
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

retrain_interval_days = 7
last_retrain_date = datetime.date.today()
n_steps_in = 60  # Use 60 time steps for input sequence
n_steps_out = 1  # Predict 1 time step ahead
n_features = 5   # Assume there are 5 features in the input data

                # List of available strategies
        
strategies = ['momentum', 'swing_trading', 'technical_analysis', 'trend_following', 'mean_reversion', 'breakout']


          



while True:
    current_date = datetime.date.today()
    
    if (current_date - last_retrain_date).days >= retrain_interval_days:
        print("Retraining LSTM models...")
          
        for symbol in symbol_list:
            data = load_preprocessed_data(symbol)
            train_lstm_model(data, f"{symbol[:-4]}_1d_lstm", n_steps_in, n_steps_out, n_features, 50, 0.2, 100, 32)
            train_lstm_model(data, f"{symbol[:-4]}_1d_lstm", n_steps_in, n_steps_out, n_features, 100, 0.3, 100, 32)
        last_retrain_date = current_date
    
    
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
        
  
        # Evaluate the performance of each strategy
        performance = {}
        for strategy_name in strategies:
            profit_loss = evaluate_strategy(strategy_name,prediction, data)
            performance[strategy_name] = profit_loss

        # Select the strategy with the highest performance
        best_strategy = max(performance, key=performance.get)

        # Use the best strategy
        strategy = trading_strategy(best_strategy, prediction, data)
        
        print(data)
        
        
        profit_loss, total_trades, successful_trades = simulate_trades(strategy)


        
        print(f'Profit/Loss: {profit_loss}')
        print(f'Total trades: {total_trades}')
        print(f'Successful trades: {successful_trades}')

        print(f"Prediction for {symbol}: {prediction}")
        
        
    
    # Execute the trading strategy
    print("Executing trading strategy...")
    execute_trading()


    # Sleep for 24 hours (86400 seconds) before checking again
    time.sleep(2)



