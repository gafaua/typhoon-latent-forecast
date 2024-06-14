import torch.nn as nn
from pytorch_tcn import TCN


class TCNForecasting(nn.Module):
    def __init__(self,
                 input_size,
                 output_size) -> None:
        super().__init__()
        self.tcn = TCN(num_inputs=input_size,
                       num_channels=[1024, 1024, 1024, 1024, 1024],
                       #output_projection=output_size,
                       causal=True,
                       use_skip_connections=True,
                       input_shape="NLC")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid(),
        )

    def forward(self, seq):
        y = self.tcn(seq)
        #print(y.shape)
        y = self.avg_pool(y.transpose(1,2)).squeeze()
        y = self.mlp(y)
        #print(y.shape)
        return y.squeeze()
