import numpy as np


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


def euclidean_distance(a, b):
    return np.sqrt((a - b) ** 2)


def dtw_1d(series1, series2):
    n = len(series1)
    m = len(series2)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(series1[i - 1], series2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    
                                          dtw_matrix[i, j - 1],    
                                          dtw_matrix[i - 1, j - 1])  

    return dtw_matrix[n, m]


def DTW(pred, true): 

    N_star, H, C_in = pred.shape
    dtw_per_sequence = []
    
    for i in range(N_star):

        dtw_per_channel = []
        for j in range(C_in):

            series_pred = pred[i, :, j]
            series_true = true[i, :, j]
            dtw_distance = dtw_1d(series_pred, series_true)
            dtw_per_channel.append(dtw_distance)
        
        dtw_per_sequence.append(np.mean(dtw_per_channel))
    
    dtw_final = np.mean(dtw_per_sequence)
    
    return dtw_final


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