import numpy as np
import torch

from tslearn.metrics import dtw, dtw_path

from loss.dilate.dilate_loss import DILATE
from loss.tildeq import tildeq_loss
from loss.varifold import TSGaussKernel, TSGaussGaussKernel, TSDotKernel, TSGaussDotKernel, time_embed, compute_position_tangent_volume, VarifoldLoss



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

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 14.7, sigma_t_2 = 1, sigma_s_2 = 14.7, n_dim = 863, device=torch.device)
    intermediate = VarifoldLoss(K, device=torch.device)

    varifold = intermediate(pred, true)

    return varifold

def VARIFOLD_metric_electricity(pred, true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 8.9, sigma_t_2 = 1, sigma_s_2 = 8.9, n_dim = 322, device=device)
    intermediate = VarifoldLoss(K, device=device)

    varifold = intermediate(pred, true)

    return varifold

def VARIFOLD_metric_exchange(pred, true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)

    K = TSGaussGaussKernel(sigma_t_1 = 1, sigma_s_1 = 1.4, sigma_t_2 = 1, sigma_s_2 = 1.4, n_dim = 9, device=device)
    intermediate = VarifoldLoss(K, device=device)

    varifold = intermediate(pred, true)

    return varifold


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


def compute_metrics(pred, true, name_of_dataset):
    
    if name_of_dataset == 'traffic.csv':
        metrics = {
            'MSE': MSE(pred, true),
            'DTW': DTW(pred, true),
            'TDI': TDI(pred, true),
            'DILATE': DILATE_08(pred, true),
            'VARIFOLD': VARIFOLD_metric_traffic(pred, true)
        }
    elif name_of_dataset == 'electricity.csv':
        metrics = {
            'MSE': MSE(pred, true),
            'DTW': DTW(pred, true),
            'TDI': TDI(pred, true),
            'DILATE': DILATE_08(pred, true),
            'VARIFOLD': VARIFOLD_metric_electricity(pred, true)
        }
    elif name_of_dataset == 'exchange_rate.csv':
        metrics = {
            'MSE': MSE(pred, true),
            'DTW': DTW(pred, true),
            'TDI': TDI(pred, true),
            'DILATE': DILATE_1(pred, true),
            'VARIFOLD': VARIFOLD_metric_exchange(pred, true)
        }
    
    return metrics