import torch
import torch.nn as nn

### Vanilla LSTM ###
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout

        self.lstm = nn.LSTM(
            input_size=self.enc_in, 
            hidden_size=self.d_model, 
            num_layers=self.e_layers, 
            batch_first=True
        )

        self.fc = nn.Linear(self.d_model, self.pred_len * self.enc_in)

        self.dropout_layer = nn.Dropout(self.dropout)

    def encoder(self, x):

        out, _ = self.lstm(x)
        
        return out 

    def forecast(self, x):
        
        out = self.encoder(x)
       
        out = out[:, -1, :]

        out = self.dropout_layer(out)

        out = self.fc(out)

        out = out.view(out.size(0), self.pred_len, self.enc_in)
        
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out
