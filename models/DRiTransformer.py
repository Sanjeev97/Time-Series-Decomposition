import torch
import torch.nn as nn
from models.iTransformer import Model as iTransformer
from layers.Decomposition import *
from layers.MLPs import RevIN, MLPfreq

class Model(nn.Module):
    """
    Decomposition + (RevIN+MLP=Trend, FAN+DLinear=Seasonal)
    
    FAN - stationary(PatchTST), non-stationary(mlpfreq)
     
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # self.output_attention = configs.output_attention
        # Decomp
        ks = configs.moving_avg
        
        if configs.decomp_method == 'moving_avg':
            # self.decomp = series_decomp(ks)
            if isinstance(ks, list):
                self.decomp = series_decomp_multi(ks)
            else:
                self.decomp = series_decomp(ks)

        if configs.decomp_method == 'dft_decomp':
            self.decomp = DFT_series_decomp(configs.top_KFD)
                             
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model_trend),
            nn.ReLU(),
            nn.Linear(configs.d_model_trend, configs.d_model_trend),
            nn.ReLU(),
            nn.Linear(configs.d_model_trend, configs.pred_len),
        )        
        
        self.revin_trend = RevIN(configs.dec_in).to(torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        ))
        
        self.decoder = iTransformer(configs)
        
        self.topK = configs.topK
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=configs.dec_in)
        
    def getdevice(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'    
    
    def forward(self, x_enc, x_mark_enc):
        # batch_size, input_len, channel
        """Takes mean along dimension 1 (input_len)
            unsqueeze(1) adds a dimension back
            repeats the mean pred_len times
            Shape transformations:

            Input: [batch_size, input_len, features]
            After mean: [batch_size, features]
            After unsqueeze: [batch_size, 1, features]

        """
        
        seasonal_init, trend_init = self.decomp(x_enc)

        seasonal_part = self.decoder(seasonal_init, x_mark_enc)  # dlinear for seasonal
        
        # trend with revin
        trend_enc = self.revin_trend(trend_init, "norm")      
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1) #rmlp for trend
        trend_out = self.revin_trend(trend_out, "denorm")
        
        dec_out = trend_out + seasonal_part
        
        return dec_out