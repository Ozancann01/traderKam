def simulate_trades(strategy, initial_balance=10000, trading_fee=0.001):
    balance = initial_balance
    asset_holdings = 0
    last_signal = 0
    total_trades = 0
    successful_trades = 0

    for index, row in strategy.iterrows():
        signal = row['Signal']
        close = row['Close']

        if signal == 1 and last_signal != 1:  # Buy
            asset_holdings = balance / (close * (1 + trading_fee))
            balance = 0
            last_signal = 1
            total_trades += 1
        elif signal == -1 and last_signal != -1:  # Sell
            new_balance = asset_holdings * close * (1 - trading_fee)
            if new_balance > balance:
                successful_trades += 1
            balance = new_balance
            asset_holdings = 0
            last_signal = -1
            total_trades += 1

    # Sell any remaining assets at the end of the simulation
    if asset_holdings > 0:
        new_balance = asset_holdings * close * (1 - trading_fee)
        if new_balance > balance:
            successful_trades += 1
        balance = new_balance

    profit_loss = balance - initial_balance
    return profit_loss, total_trades, successful_trades
