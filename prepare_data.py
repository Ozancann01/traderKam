import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from strategies import (
    trend_following, mean_reversion, breakout, 
    momentum, swing_trading, technical_analysis
)

def prepare_data(data, lookahead=1):
    # Calculate future returns
    data['FutureReturn'] = data['Close'].shift(-lookahead).pct_change(lookahead)

    # Apply strategies
    data['TF'] = trend_following(data.copy())
    data['MR'] = mean_reversion(data.copy())
    data['BO'] = breakout(data.copy())
    data['MT'] = momentum(data.copy())
    data['ST'] = swing_trading(data.copy())
    data['TA'] = technical_analysis(data.copy())

    # Remove the first few rows without signals
    data.dropna(inplace=True)

    # Define target labels
    data['Target'] = np.where(data['FutureReturn'] > 0, 1, 0)

    return data

# Load your data here, for example:
data = pd.read_csv('data/BTCUSDT_1D.csv')

# Prepare the data
prepared_data = prepare_data(data)



def train_meta_strategy(data):
    # Define features and target variable
    X = data[['TF', 'MR', 'BO', 'MT', 'ST', 'TA']]
    y = data['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

# Train the meta-strategy
model = train_meta_strategy(prepared_data)

