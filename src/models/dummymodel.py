
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        
        return x
    