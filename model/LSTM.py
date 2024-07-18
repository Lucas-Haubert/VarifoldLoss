import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        self.lstm = nn.LSTM(
            input_size=self.enc_in, 
            hidden_size=self.d_model, 
            num_layers=self.e_layers, 
            batch_first=True
        )

        self.fc = nn.Linear(self.d_model, self.pred_len * self.enc_in)

    def encoder(self, x):

        # x has size [B, seq_len, C]

        h0 = torch.zeros(self.e_layers, x.size(0), self.d_model, device=x.device)
        c0 = torch.zeros(self.e_layers, x.size(0), self.d_model, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        #print("shape of the lstm output: ", out.shape) # [B, seq_len, d_model]
        
        return out 

    def forecast(self, x):
        
        out = self.encoder(x)
       
        out = out[:, -1, :]
        #print("shape of lstm output after sclicing", out.shape)

        out = self.fc(out)
        #print("shape of the output after fc layer: ", out.shape)

        out = out.view(out.size(0), self.pred_len, self.enc_in)
        #print("shape of the output after reshaping: ", out.shape)
        
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out
