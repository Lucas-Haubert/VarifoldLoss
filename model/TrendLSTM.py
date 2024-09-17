import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

# Adaptation of the LSTM model to include the trend decomposition (Autoformer)

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout

        self.decompsition = series_decomp(configs.moving_avg)

        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True
        )

        self.fc_lstm = nn.Linear(self.d_model, self.pred_len * self.enc_in)
        self.dropout_layer = nn.Dropout(self.dropout)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)

        trend_init = trend_init.permute(0, 2, 1)  # [B, D, seq_len] -> [B, seq_len, D]
        trend_out = self.Linear_Trend(trend_init)
        trend_out = trend_out.permute(0, 2, 1)  # [B, pred_len, D] -> [B, D, pred_len]

        seasonal_out, _ = self.lstm(seasonal_init)
        seasonal_out = seasonal_out[:, -1, :]  
        seasonal_out = self.dropout_layer(seasonal_out)
        seasonal_out = self.fc_lstm(seasonal_out)
        seasonal_out = seasonal_out.view(seasonal_out.size(0), self.pred_len, self.enc_in)

        output = trend_out + seasonal_out 

        return output 

    def forecast(self, x):
        return self.encoder(x)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
