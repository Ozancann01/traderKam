import pandas as pd
import ta

def load_data(filename):
    return pd.read_csv(f"data/{filename}")

def calculate_technical_indicators(data):
    # Add Simple Moving Average (SMA)
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=14).sma_indicator()

    # Add Exponential Moving Average (EMA)
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=14).ema_indicator()

    # Add Relative Strength Index (RSI)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    # Add Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()

    # Add Average True Range (ATR)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

    # Add Bollinger Band Width
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()

    # Add Standard Deviation
    data['std_dev'] = data['Close'].rolling(window=14).std()

    # Add Parabolic SAR
    data['SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()

    # Add Average Directional Movement Index (ADX)
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()

    return data

def preprocess_data(filename):
    data = load_data(filename)
    data_with_indicators = calculate_technical_indicators(data)
    return data_with_indicators.dropna()

# Preprocess data for each cryptocurrency pair
symbol_list = ['BTCUSDT_1d.csv', 'ETHUSDT_1d.csv', 'BNBUSDT_1d.csv', 'XRPUSDT_1d.csv']

for symbol in symbol_list:
    preprocessed_data = preprocess_data(symbol)
    preprocessed_data.to_csv(f"data/preprocessed_{symbol}", index=False)
    print(f"Preprocessed {symbol} and saved to data/preprocessed_{symbol}")