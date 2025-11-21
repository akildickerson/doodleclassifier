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
            nn.Conv2d(in_channels, conv_channels[0], 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_channels[0], conv_channels[1], 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        logits = self.layers(x)
        return logits
