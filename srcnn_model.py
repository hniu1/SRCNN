# srcnn_model.py
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    PyTorch version of your TF SRCNN:
    Conv2D(64, 9x9) → PReLU → Conv2D(32, 1x1) → PReLU → Conv2D(1, 5x5)
    """
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.PReLU()
        )
        self.layer3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
