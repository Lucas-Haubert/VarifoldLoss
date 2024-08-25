import numpy as np
import pandas as pd
import torch

from tslearn.metrics import dtw, dtw_path
from scipy.fft import rfft, rfftfreq


# Definition of the metrics

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


def TDI(pred, true):
    N, H, C = pred.shape
    total_tdi_distance = 0
    for i in range(N):
        pred_sequence = pred[i]
        true_sequence = true[i] 
        path, _ = dtw_path(pred_sequence, true_sequence)
        tdi_distance = sum(abs(i-j) for (i, j) in path)
        total_tdi_distance += tdi_distance
    return total_tdi_distance / N


def fourier_spectra(series, frequency_range):
    fourier_coefficients = rfft(series, axis=1)
    frequencies = rfftfreq(series.shape[1])
    freq_indices = np.where((frequencies >= frequency_range[0]) & (frequencies < frequency_range[1]))[0]
    filtered_fourier_coefs = np.abs(fourier_coefficients)[:,freq_indices,:]
    mean_fourier_coefs_axis_1 = filtered_fourier_coefs.mean(axis=1)
    mean_fourier_coefs = mean_fourier_coefs_axis_1.mean()
    return mean_fourier_coefs

def rFFT(series_pred, series_ground, frequency_range):
    mean_fourier_coef_pred = fourier_spectra(series_pred, frequency_range)
    mean_fourier_coef_ground = fourier_spectra(series_ground, frequency_range)
    metric = (mean_fourier_coef_pred - mean_fourier_coef_ground) / mean_fourier_coef_ground
    return metric

def calculate_spectral_entropy(series):
    N = len(series[1])
    yf = rfft(series, axis=1)
    power_spectrum = 2.0/N * np.abs(yf[:, :N//2, :])
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum, axis=1, keepdims=True)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12), axis=1)
    mean_spectral_entropy = np.mean(spectral_entropy)
    return mean_spectral_entropy

def rSE(series_pred, series_ground):
    spectral_entropy_pred = calculate_spectral_entropy(series_pred)
    spectral_entropy_ground = calculate_spectral_entropy(series_ground)
    metric = (spectral_entropy_pred - spectral_entropy_ground) / spectral_entropy_ground
    return metric


# Compute the metrics

def compute_metrics(pred, true, list_of_metrics):
    metrics = {}
    
    for metric in list_of_metrics:
        if metric.startswith('rFFT_'):
            freq_range_str = metric[len('rFFT_') + 1:-1]
            first, last = freq_range_str.split('_')
            first = float(first.replace('dot', '.'))
            last = float(last.replace('dot', '.'))
            metrics[metric] = rFFT(pred, true, frequency_range=(first, last))
        else:
            metrics[metric] = globals()[metric](pred, true)
    
    return metrics

# Cette version permet de calculer les métriques depuis list_of_metrics directement, pas comme la version suivante

# def compute_metrics(pred, true):
    
#     metrics = {
#             'MSE': MSE(pred, true),
#             'MAE': MAE(pred, true),
#             'DTW': DTW(pred, true),
#             'TDI': TDI(pred, true),
#             'rFFT_low': rFFT(pred, true, frequency_range=(0, 0.02)),
#             'rFFT_mid': rFFT(pred, true, frequency_range=(0.02, 0.15)),
#             'rFFT_high': rFFT(pred, true, frequency_range=(0.15, 0.35)),
#             'rSE': rSE(pred, true)
#     }
    
#     return metrics



# Mean, median and std for run.py

def compute_mean_median_std_metrics(metrics_list):
    metric_names = metrics_list[0].keys()
    aggregated_metrics = {metric: [] for metric in metric_names}

    for metrics in metrics_list:
        for metric in metric_names:
            aggregated_metrics[metric].append(metrics[metric])
            
    mean_metrics = {metric: np.mean(aggregated_metrics[metric]) for metric in metric_names}
    median_metrics = {metric: np.median(aggregated_metrics[metric]) for metric in metric_names}
    std_metrics = {metric: np.std(aggregated_metrics[metric]) for metric in metric_names}

    return mean_metrics, median_metrics, std_metrics