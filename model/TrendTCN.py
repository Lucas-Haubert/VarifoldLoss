import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from layers.Autoformer_EncDec import series_decomp

# Adaptation of the TCN model to include the trend decomposition (Autoformer)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.num_channels = [configs.out_dim_first_layer * (2 ** i) for i in range(configs.e_layers)]
        self.kernel_size = configs.fixed_kernel_size_tcn
        self.dropout = configs.dropout

        self.decompsition = series_decomp(configs.moving_avg)

        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        self.seasonal_tcn = TemporalConvNet(self.enc_in, self.num_channels, kernel_size=self.kernel_size, dropout=self.dropout)
        self.seasonal_linear = nn.Linear(self.num_channels[-1], self.enc_in)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)

        trend_init = trend_init.permute(0, 2, 1)  # [B, D, seq_len] -> [B, seq_len, D]
        trend_out = self.Linear_Trend(trend_init)
        trend_out = trend_out.permute(0, 2, 1)  # [B, pred_len, D] -> [B, D, pred_len]

        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, seq_len, D] -> [B, D, seq_len]
        seasonal_out = self.seasonal_tcn(seasonal_init)
        seasonal_out = seasonal_out[:, :, -self.pred_len:]  
        seasonal_out = seasonal_out.permute(0, 2, 1)  # [B, num_channels[-1], pred_len] -> [B, pred_len, num_channels[-1]]
        seasonal_out = self.seasonal_linear(seasonal_out)  # [B, pred_len, enc_in]

        output = trend_out + seasonal_out
        
        return output  

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out  # [B, pred_len, D]

