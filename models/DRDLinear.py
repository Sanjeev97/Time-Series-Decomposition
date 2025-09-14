import torch
import torch.nn as nn
from models.DLinear import Model as DLinear
# from DLinear import Model as DLinear

from layers.MLPs import series_decomp, RevIN


class Model(nn.Module):
    """
    Decomposition + (RevIN+MLP=Trend, FAN+DLinear=Seasonal)
    
    FAN - stationary(DLinear), non-stationary(mlpfreq)
     
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # self.output_attention = configs.output_attention

        # Decomp
        ks = configs.moving_avg
        
        self.decomp = series_decomp(ks)
                             
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model_trend),
            nn.ReLU(),
            nn.Linear(configs.d_model_trend, configs.d_model_trend),
            nn.ReLU(),
            nn.Linear(configs.d_model_trend, configs.pred_len),
        )        
        
        self.revin_trend = RevIN(configs.dec_in)
        # self.revin_trend = RevIN(configs.dec_in).to(torch.device(
        #                     "cuda" if torch.cuda.is_available() else "cpu"
        #                 ))        
        
        self.decoder = DLinear(configs)
        
    def getdevice(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'    
    
    def forward(self, x_enc):
        # batch_size, input_len, channel
        """Takes mean along dimension 1 (input_len)
            unsqueeze(1) adds a dimension back
            repeats the mean pred_len times
            Shape transformations:

            Input: [batch_size, input_len, features]
            After mean: [batch_size, features]
            After unsqueeze: [batch_size, 1, features]

        """
        
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        
        seasonal_init, trend_init = self.decomp(x_enc)

        seasonal_part = self.decoder(seasonal_init)  # dlinear for seasonal
        
        # trend with revin
        trend_enc = self.revin_trend(trend_init, "norm")      
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1) #rmlp for trend
        trend_out = self.revin_trend(trend_out, "denorm")
        
        dec_out = trend_out + seasonal_part
        
        return dec_out
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DLinear')
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=4, help='output sequence length')
    parser.add_argument('--dec_in', type=int, default=1, help='number of input channels')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--d_model_trend', type=int, default=512, help='dimension of model')
    args = parser.parse_args()

    model = Model(args)
    x = torch.randn(1, 4, 512)
    y = model(x)
    print(y.shape) # [32, 96, 1]
        
