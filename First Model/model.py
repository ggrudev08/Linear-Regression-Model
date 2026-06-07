import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    
    def forward(self, x):
        return self.linear(x).squeeze(1)


    