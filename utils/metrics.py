import numpy as np
import pandas as pd
import torch

from tslearn.metrics import dtw, dtw_path

from loss.dilate.dilate_loss import DILATE
from loss.tildeq import tildeq_loss
from loss.varifold import TSGaussKernel, TSGaussGaussKernel, TSDotKernel, TSGaussDotKernel, time_embed, compute_position_tangent_volume, VarifoldLoss

from scipy.fft import rfft, rfftfreq

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


def DILATE_metric(pred, true, alpha):

    pred = torch.from_numpy(pred).cpu()
    true = torch.from_numpy(true).cpu()

    dilate = DILATE(pred, true, alpha=alpha)

    return dilate


def VARIFOLD_metric_traffic(pred, true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 14.7, sigma_t_2 = 1, sigma_s_2 = 14.7, n_dim = 863, device=device)
    intermediate = VarifoldLoss(K, device=device)

    # Without introducing a new batch_size
    #varifold = intermediate(pred, true)

    #new_batch_size = 8
    new_batch_size = 4
    initial_number_of_batches = pred.shape[0]
    varifold_total = 0

    for i in range(0, initial_number_of_batches, new_batch_size):
        pred_batch = pred[i:i+new_batch_size]
        true_batch = true[i:i+new_batch_size]
        varifold_batch = intermediate(pred_batch, true_batch)
        varifold_total += varifold_batch * new_batch_size
    
    varifold = varifold_total / (initial_number_of_batches // new_batch_size)

    return varifold.cpu()

def VARIFOLD_metric_electricity(pred, true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 8.9, sigma_t_2 = 1, sigma_s_2 = 8.9, n_dim = 322, device=device)
    intermediate = VarifoldLoss(K, device=device)

    # Without introducing a new batch_size
    #varifold = intermediate(pred, true)

    #new_batch_size = 8
    new_batch_size = 4
    initial_number_of_batches = pred.shape[0]
    varifold_total = 0

    for i in range(0, initial_number_of_batches, new_batch_size):
        pred_batch = pred[i:i+new_batch_size]
        true_batch = true[i:i+new_batch_size]
        varifold_batch = intermediate(pred_batch, true_batch)
        varifold_total += varifold_batch * new_batch_size
    
    varifold = varifold_total / (initial_number_of_batches // new_batch_size)

    return varifold.cpu()

def VARIFOLD_metric_exchange(pred, true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 1.4, sigma_t_2 = 1, sigma_s_2 = 1.4, n_dim = 9, device=device)
    intermediate = VarifoldLoss(K, device=device)

    varifold = intermediate(pred, true)

    return varifold.cpu()


def DILATE_05(pred, true):
    return DILATE_metric(pred, true, alpha=0.5)

def DILATE_08(pred, true):
    return DILATE_metric(pred, true, alpha=0.8)

def DILATE_1(pred, true):  
    return DILATE_metric(pred, true, alpha=1)


def softDTW(pred, true):
    return DILATE_metric(pred, true, alpha=1)


def softTDI(pred, true):
    return DILATE_metric(pred, true, alpha=0)


def TILDEQ_metric(pred, true, alpha):

    pred = torch.from_numpy(pred).cpu()
    true = torch.from_numpy(true).cpu()

    tildeq = tildeq_loss(pred, true, alpha=alpha)

    return tildeq


def TILDEQ_05(pred, true):
    return TILDEQ_metric(pred, true, alpha=0.5)


def TILDEQ_1(pred, true):
    return TILDEQ_metric(pred, true, alpha=1)


def TILDEQ_00(pred, true):
    return TILDEQ_metric(pred, true, alpha=0)

# def fourier_spectra(series, frequency_range):
    
#     fourier_coefficients = rfft(series, axis=1)
#     print("fourier_coefficients", fourier_coefficients)
#     print("fourier_coefficients.shape", fourier_coefficients.shape)
#     frequencies = rfftfreq(len(series[1]))
#     print("frequencies", frequencies)
#     print("frequencies.shape", frequencies.shape)
    
#     fourier_df = pd.DataFrame({'frequency': frequencies, 'amplitude': np.abs(fourier_coefficients)})
    
#     frequency_band = fourier_df[(fourier_df['frequency'] >= frequency_range[0]) & (fourier_df['frequency'] < frequency_range[1])]
#     mean_fourier_coef = frequency_band['amplitude'].mean()
    
#     return mean_fourier_coef

def fourier_spectra(series, frequency_range):
    
    fourier_coefficients = rfft(series, axis=1)
    #print("fourier_coefficients.shape", fourier_coefficients.shape)
    
    frequencies = rfftfreq(series.shape[1])
    #print("frequencies.shape", frequencies.shape)
    
    freq_indices = np.where((frequencies >= frequency_range[0]) & (frequencies < frequency_range[1]))[0]
    #print("freq_indices.shape", freq_indices.shape)
    
    filtered_fourier_coefs = np.abs(fourier_coefficients)[:,freq_indices,:]
    #print("filtered_fourier_coefs.shape", filtered_fourier_coefs.shape)
    
    mean_fourier_coefs_axis_1 = filtered_fourier_coefs.mean(axis=1)
    #print("mean_fourier_coefs_axis_1.shape", mean_fourier_coefs_axis_1.shape)

    mean_fourier_coefs = mean_fourier_coefs_axis_1.mean()
    #print("mean_fourier_coefs.shape", mean_fourier_coefs.shape)

    return mean_fourier_coefs

def rFFT(series_pred, series_ground, frequency_range):
    mean_fourier_coef_pred = fourier_spectra(series_pred, frequency_range)
    mean_fourier_coef_ground = fourier_spectra(series_ground, frequency_range)

    metric = (mean_fourier_coef_pred - mean_fourier_coef_ground) / mean_fourier_coef_ground

    return metric

def calculate_spectral_entropy(series):

    N = len(series[1])
    #print("N", N)
    yf = rfft(series, axis=1)
    #print("yf.shape", yf.shape)
    power_spectrum = 2.0/N * np.abs(yf[:, :N//2, :])
    #print("power_spectrum.shape", power_spectrum.shape)
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum, axis=1, keepdims=True)
    #print("power_spectrum_norm.shape", power_spectrum_norm.shape)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12), axis=1)
    #print("spectral_entropy.shape", spectral_entropy.shape)
    mean_spectral_entropy = np.mean(spectral_entropy)
    #print("mean_spectral_entropy", mean_spectral_entropy)
    
    return mean_spectral_entropy

def rSE(series_pred, series_ground):
    spectral_entropy_pred = calculate_spectral_entropy(series_pred)
    spectral_entropy_ground = calculate_spectral_entropy(series_ground)

    metric = (spectral_entropy_pred - spectral_entropy_ground) / spectral_entropy_ground

    return metric


def compute_metrics(pred, true):
    
    metrics = {
            'MSE': MSE(pred, true),
            'MAE': MAE(pred, true),
            'DTW': DTW(pred, true),
            'rFFT_low': rFFT(pred, true, frequency_range=(0, 0.05)),
            'rFFT_mid': rFFT(pred, true, frequency_range=(0.05, 0.25)),
            'rFFT_high': rFFT(pred, true, frequency_range=(0.25, 0.5)),
            'rSE': rSE(pred, true)
    }
    
    return metrics