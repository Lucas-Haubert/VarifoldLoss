import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.proj_dim_1 = nn.Linear(self.enc_in, self.d_model)
        self.proj_dim_2 = nn.Linear(self.d_model, self.enc_in)

        self.proj_time_1 = nn.Linear(self.seq_len, self.d_model)
        self.proj_time_2 = nn.Linear(self.d_model, self.pred_len)

        self.dropout_layer = nn.Dropout(self.dropout)

    def encoder(self, x):
    
        x = x.permute(0, 2, 1) # [B, seq_len, D] -> [B, D, seq_len]
        
        x = self.proj_time_1(x) # [B, D, seq_len] -> [B, D, d_model]
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.proj_time_2(x) # [B, D, d_model] -> [B, D, pred_len]
        
        x = x.permute(0, 2, 1) # [B, D, pred_len] -> [B, pred_len, D]

        x = self.proj_dim_1(x) # [B, pred_len, D] -> [B, pred_len, d_model]
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.proj_dim_2(x) # [B, pred_len, d_model] -> [B, pred_len, D]

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out  # [B, pred_len, D]