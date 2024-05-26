import numpy as np
from tslearn.metrics import dtw


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1) 


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def RMAE(pred, true):
    return np.sqrt(MAE(pred, true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return 100 * np.mean(np.abs((pred - true) / true))


def SMAPE(pred, true):
    return 100 * np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def normalize(array):
    mean = np.mean(array, axis=1, keepdims=True)
    std = np.std(array, axis=1, keepdims=True)
    return (array - mean) / std


def nMAE(pred, true):
    pred_normalized = normalize(pred)
    true_normalized = normalize(true)
    return MAE(pred_normalized, true_normalized)


def nMSE(pred, true):
    pred_normalized = normalize(pred)
    true_normalized = normalize(true)
    return MSE(pred_normalized, true_normalized)


def nRMAE(pred, true):
    return np.sqrt(nMAE(pred, true))


def nRMSE(pred, true):
    return np.sqrt(nMSE(pred, true))


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


def nDTW(pred, true):
    pred_normalized = normalize(pred)
    true_normalized = normalize(true)
    return DTW(pred_normalized, true_normalized)


def metric(pred, true):
    metrics = {
        'RSE': RSE(pred, true),
        'CORR': CORR(pred, true),
        'MAE': MAE(pred, true),
        'RMAE': RMAE(pred, true),
        'MSE': MSE(pred, true),
        'RMSE': RMSE(pred, true),
        'MAPE': MAPE(pred, true),
        'SMAPE': SMAPE(pred, true),
        'MSPE': MSPE(pred, true),
        'nMAE': nMAE(pred, true),
        'nMSE': nMSE(pred, true),
        'nRMAE': nRMAE(pred, true),
        'nRMSE': nRMSE(pred, true),
        'MASE': MASE(pred, true),
        'DTW': DTW(pred, true), 
        'nDTW': nDTW(pred, true)
    }
    return metrics