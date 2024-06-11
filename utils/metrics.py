import numpy as np
from tslearn.metrics import dtw


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def RMAE(pred, true):
    return np.sqrt(MAE(pred, true))


def MAPE(pred, true):
    return 100 * np.mean(np.abs((pred - true) / true))


def SMAPE(pred, true):
    return 100 * np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1) 


def MASE(pred, true):
    mae = np.mean(np.abs(true - pred), axis=1)
    naive_forecast = np.mean(np.abs(np.diff(true, axis=1)), axis=1)
    return np.mean(mae / naive_forecast)


def DTW(pred, true):
    N, H, C = pred.shape
    total_dtw_distance = 0

    for i in range(N):
        pred_sequence = pred[i]
        true_sequence = true[i] 
        distance = dtw(pred_sequence, true_sequence)
        total_dtw_distance += distance

    return total_dtw_distance / N   


# Apply this function to pred and true before passing them into one of the metrics below
# This approach normalizes each sequence based on their proper statistics, while sk-learn scaler
# normalizes relatively to the train dataset
# => Different approach => Use later if needed
def normalize(array):
    mean = np.mean(array, axis=1, keepdims=True)
    std = np.std(array, axis=1, keepdims=True)
    return (array - mean) / std


def metric(pred, true):
    metrics = {
        'MSE': MSE(pred, true),
        'MAE': MAE(pred, true),
        'RMSE': RMSE(pred, true),
        'RMAE': RMAE(pred, true),
        'MAPE': MAPE(pred, true),
        'SMAPE': SMAPE(pred, true),
        'MSPE': MSPE(pred, true),
        'RSE': RSE(pred, true),
        'CORR': CORR(pred, true),
        'MASE': MASE(pred, true),
        'DTW': DTW(pred, true),
    }
    return metrics