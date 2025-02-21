
import torch
import torch.nn as nn
import torch.nn.functional as F



# loss for classification
class LossCE(nn.Module):
    def __init__(self):
        super(LossCE, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none')
        
    
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        return loss


class ComputeLoss:
    
    def __init__(self):
        super().__init__()
        


class DummyModel(nn.Module):
    
    def __init__(self, num_classes=2):
        super(DummyModel, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.detect_head = nn.Sequential(
            nn.Conv2d(64, num_classes + 4, kernel_size=1)  # (num_classes for classification + 4 for bbox)
        )
    
    def forward(self, x):
        
        x = self.backbone(x)
        x = self.detect_head(x)
        
        
        
        return x




if __name__ == "__main__":
    # generate example im data 3x640x640
    
    input = torch.randn(1, 3, 640, 640)
    
    model = DummyModel(2)
    print(model)
    
    output = model(input)
    print(output.shape)
        
    
    