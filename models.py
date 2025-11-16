import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__(self)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, 14)
        )

    def forward(self, x) -> torch.Tensor:
        logits = self.layers(x)
        return logits