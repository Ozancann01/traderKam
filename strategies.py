import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator


# Add these imports at the beginning of your strategies.py file
from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator

# Momentum Strategy
def momentum(data, period=14, threshold=0.05):
    # Calculate the rate of change
    data['ROC'] = data['Close'].pct_change(period)

    # Generate signals
    data['Signal'] = np.where(data['ROC'] > threshold, 1,
                              np.where(data['ROC'] < -threshold, -1, 0))

    return data['Signal']

# Swing Trading Strategy
def swing_trading(data, period=14):
    # Identify local minima and maxima using argrelextrema
    minima = argrelextrema(data['Close'].values, np.less_equal, order=period)
    maxima = argrelextrema(data['Close'].values, np.greater_equal, order=period)

    # Initialize signals
    data['Signal'] = 0

    # Generate buy signals at local minima
    data.loc[data.index[minima], 'Signal'] = 1

    # Generate sell signals at local maxima
    data.loc[data.index[maxima], 'Signal'] = -1

    return data['Signal']

# Technical Analysis Strategy
def technical_analysis(data, period=14):
    # Calculate moving averages
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['EMA'] = data['Close'].ewm(span=period).mean()

    # Calculate RSI
    rsi_indicator = RSIIndicator(data['Close'], window=period)
    data['RSI'] = rsi_indicator.rsi()

    # Generate signals based on moving averages and RSI
    data['Signal'] = np.where((data['Close'] > data['SMA']) & (data['Close'] > data['EMA']) & (data['RSI'] < 30), 1,
                              np.where((data['Close'] < data['SMA']) & (data['Close'] < data['EMA']) & (data['RSI'] > 70), -1, 0))

    return data['Signal']


def trend_following(data, period=14):
    # Calculate the moving average
    data['MA'] = data['Close'].rolling(window=period).mean()
    
    # Generate signals
    data['Signal'] = np.where(data['Close'] > data['MA'], 1, -1)
    
    return data['Signal']

def mean_reversion(data, period=14):
    # Calculate the moving average and standard deviation
    data['MA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()

    # Generate signals
    data['Signal'] = np.where(data['Close'] < (data['MA'] - 2 * data['STD']), 1,
                              np.where(data['Close'] > (data['MA'] + 2 * data['STD']), -1, 0))
    
    return data['Signal']

def breakout(data, period=14):
    # Calculate the highest high and lowest low of the given period
    data['Highest'] = data['High'].rolling(window=period).max()
    data['Lowest'] = data['Low'].rolling(window=period).min()

    # Generate signals
    data['Signal'] = np.where(data['Close'] > data['Highest'], 1,
                              np.where(data['Close'] < data['Lowest'], -1, 0))

    return data['Signal']
