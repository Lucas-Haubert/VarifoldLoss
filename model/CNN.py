import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

### Vanilla cnn ###
class ConvNet(nn.Module):
    def __init__(self, enc_in, d_model):
        super(ConvNet, self).__init__()
    
        self.kernel_size = 3
        self.padding = 1 # self.kernel//2
        self.d_model = d_model

        self.layer1 = nn.Sequential(
                nn.Conv1d(enc_in, 64, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(64),
                nn.ReLU()
                )
                
        # self.layer2 = nn.Sequential(
        #         nn.Conv1d(64, 128, kernel_size=self.kernel_size, padding=self.padding),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU()
        #         )
        
        self.layer3 = nn.Sequential(
                nn.Conv1d(64, self.d_model, kernel_size=self.kernel_size, padding=self.padding),
                nn.ReLU()
                )
                          
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = self.layer3(out)

        return out

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.model = ConvNet(self.enc_in, self.d_model)
        
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(self.d_model, configs.enc_in, bias=True) 
            
    def forecast(self, x_enc):
        # x_enc is [B, seq_len, enc_in] originally
        x_enc = x_enc.permute(0,2,1) # [B, enc_in, seq_len]
        enc_out = self.model(x_enc) # [B, d_model, seq_len]  

        dec_out = self.predict_linear(enc_out) # [B, d_model, seq_len + pred_len] 

        dec_out = dec_out.permute(0,2,1) # [B, seq_len + pred_len, d_model]
        dec_out = self.projection(dec_out) # [B, seq_len + pred_len, enc_in]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, enc_in]