from sklearn.metrics import r2_score
import numpy as np
from catboost import CatBoostRegressor

SEED = 42

def fit_model(X_train, y_train):
    model = CatBoostRegressor(
        iterations=150,
        depth=7,
        l2_leaf_reg=1,
        learning_rate=0.1,
        verbose=False,
        random_seed=SEED,
    )
    model.fit(X_train, y_train)
    return model


def predict_using_model(model, X_val):
    return np.exp(model.predict(X_val)) - 1


def r2_score_custom(y_true, y_pred):
    y_true = np.exp(y_true) - 1
    return r2_score(y_true, y_pred)


def rmsle_custom(y_true, y_pred):
    # Ensure there are no negative values in predictions and actuals
    y_true = np.maximum(0, np.exp(y_true) - 1)
    y_pred = np.maximum(0, y_pred)

    # Calculate the logarithm of predictions and actuals
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Compute the squared differences of the logs
    squared_log_errors = (log_pred - log_true) ** 2

    # Calculate the mean of the squared log errors
    mean_squared_log_errors = np.mean(squared_log_errors)

    # Take the square root of the mean squared log errors
    rmsle_value = np.sqrt(mean_squared_log_errors)
    return rmsle_value
