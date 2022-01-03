import numpy as np


def split_dataset_into_X_y(dataset, lag_size: 'int > 0' = 15):
    """Split the dataset into X and y according to the number of lags.
    The default lag size is set to 15.
    """
    X, y = list(), list()
    for i in range(lag_size, dataset.shape[0]):
        X.append(dataset[i - lag_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


def mape(actual, prediction):
    """Mean Absolute Percentage Error"""
    actual, prediction = np.array(actual), np.array(prediction)
    return np.mean(np.abs((prediction - actual) / actual)) * 100


def rmse(actual, prediction):
    """Root-mean-square-deviation"""
    from sklearn.metrics import mean_squared_error
    actual, prediction = np.array(actual), np.array(prediction)
    mse = mean_squared_error(actual, prediction)
    return np.sqrt(mse)


def cv(prediction):
    """Coefficient of Variation"""
    prediction = np.array(prediction)
    return np.std(prediction, ddof=1) / np.mean(prediction) * 100


def mkdir_if_not_exists(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
