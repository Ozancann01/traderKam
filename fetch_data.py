import ccxt
import pandas as pd

def fetch_ohlcv_data(exchange, symbol, timeframe, since=None, limit=None):
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    headers = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    return pd.DataFrame(ohlcv_data, columns=headers)

def save_to_csv(data, symbol, filename):
    data.to_csv(f"data/{filename}", index=False)
    print(f"Saved {symbol} data to data/{filename}")


# Initialize Binance exchange object
exchange = ccxt.binance({
    'apiKey': 'yl2oprSsxWnOUt5emls7AVIaYoGWRXWddtPk0WiNVns4SKvlI5nyyMtyjAwxCSs6',         # Replace with your Binance API key
    'secret': 'KhiklzjukTeqcFSMKm1LMbLjVzIF77kuGaigyGaecPlkRqnabqC8HFognlqR6BvU',      # Replace with your Binance secret key
    'enableRateLimit': True,
})

# Define cryptocurrency pairs and timeframe
symbol_list = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT']
timeframe = '1d'  # Daily data

# Fetch historical price data and save to CSV files
for symbol in symbol_list:
    data = fetch_ohlcv_data(exchange, symbol, timeframe)
    save_to_csv(data, symbol, f"{symbol.replace('/', '')}_{timeframe}.csv")
