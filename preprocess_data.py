import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def rate_of_change_and_difference(data, indicators=['SMA', 'EMA', 'RSI', 'MACD']):
    for indicator in indicators:
        data[f'{indicator}_ROC'] = data[indicator].pct_change()
        data[f'{indicator}_Diff'] = data[indicator].diff()

    return data

data = pd.read_csv("data/preprocessed_BTCUSDT_1d.csv")
print(data.isna().sum())


def preprocess_data(symbol):
    data = load_data(symbol)
    data_with_indicators = calculate_technical_indicators(data)
    data_with_moving_averages = moving_averages(data_with_indicators)
    data_with_roc_and_diff = rate_of_change_and_difference(data_with_moving_averages)
    data_with_roc_and_diff = data_with_roc_and_diff.dropna()

    scaled_data = scale_data(data_with_roc_and_diff)

    return scaled_data



def moving_averages(data, windows=[5, 10, 20]):
    for window in windows:
        data[f'SMA_{window}'] = data['SMA'].rolling(window=window).mean()
        data[f'EMA_{window}'] = data['EMA'].ewm(span=window).mean()
        data[f'RSI_{window}'] = data['RSI'].rolling(window=window).mean()
        data[f'MACD_{window}'] = data['MACD'].rolling(window=window).mean()

    return data

def scale_data(data):
    scaler = MinMaxScaler()

    # List the columns you want to scale
    columns_to_scale = [
        'SMA', 'EMA', 'RSI', 'MACD',
        'SMA_ROC', 'EMA_ROC', 'RSI_ROC', 'MACD_ROC',
        'SMA_Diff', 'EMA_Diff', 'RSI_Diff', 'MACD_Diff'
    ]

    scaled_data = data.copy()
    scaled_data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return scaled_data



# Preprocess data for each cryptocurrency pair
symbol_list = ['BTCUSDT_1d.csv', 'ETHUSDT_1d.csv', 'BNBUSDT_1d.csv', 'XRPUSDT_1d.csv']

for symbol in symbol_list:
    preprocessed_data = preprocess_data(symbol)
    preprocessed_data.to_csv(f"data/preprocessed_{symbol}", index=False)
    print(f"Preprocessed {symbol} and saved to data/preprocessed_{symbol}")