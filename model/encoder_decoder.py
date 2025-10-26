import torch
import torch.nn.functional as F
from torch import nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,x):
        y=self.encoder(x)
        x_hat=self.decoder(y)
        return x_hat
    def get_name(self):
        return self.encoder.__class__.__name__+'_'+self.decoder.__class__.__name__
