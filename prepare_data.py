import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Train the models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
    }

    # Set up the hyperparameter grid for RandomForest
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Perform grid search with cross-validation for RandomForest
    grid_search = GridSearchCV(estimator=models['RandomForest'], param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Retrieve the best model for RandomForest
    models['RandomForest'] = grid_search.best_estimator_

    # Fit the Logistic Regression and SVM models
    models['LogisticRegression'].fit(X_train, y_train)
    models['SVM'].fit(X_train, y_train)

    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        print(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        if name == 'RandomForest':
            print("Best Parameters for RandomForest:", grid_search.best_params_)

# Train the meta-strategies
train_meta_strategy(prepared_data)

