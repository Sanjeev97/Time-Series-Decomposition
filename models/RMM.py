import torch
import torch.nn as nn
from layers.Decomposition import *
from layers.MLPs import MLPfreq, RevIN

class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

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
        
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=configs.dec_in)   
    
    def forward(self, x_enc):#b,l,c
        # seasonal_init, trend_init = main_freq_part(x_enc, k=self.topK)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        seasonal_part = self.trend(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        # seasonal_part = self.model_freq(seasonal_init.transpose(1,2), x_enc.transpose(1,2)).transpose(1,2) 
        
        # trend with revin
        trend_enc = self.revin_trend(trend_init, "norm")      
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, "denorm")
        
        dec_out = trend_out  + seasonal_part 
        
        return dec_out    
          
