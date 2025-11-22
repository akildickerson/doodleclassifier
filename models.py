import torch
import torch.nn as nn
from config import CONFIG


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = CONFIG["model"]
        in_channels = model["in_channels"]
        conv_channels = model["conv_channels"]
        hidden_dims = model["hidden_dims"]
        num_classes = model["num_classes"]

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], 3, 1), # 16 x 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_channels[0], conv_channels[1], 3, 1), # 32 x 11 x 11
            nn.Conv2d(conv_channels[1], conv_channels[1], 3, 1), # 32 x 9 x 9
            nn.Conv2d(conv_channels[1], conv_channels[1], 3, 1), # 32 x 7 x 7
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x 3 x 3
            nn.Flatten(), # 32 x 9
            nn.Linear(hidden_dims[0], hidden_dims[1]), # 288 x 128
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes), # 128 x 14
        )

    def forward(self, x) -> torch.Tensor:
        logits = self.layers(x)
        return logits
