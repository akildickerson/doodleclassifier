CONFIG = {
    "model": {
        "in_channels": 1,
        "conv_channels": [16, 32, 64],
        "hidden_dims": [64, 128, 256],
        "num_classes": 14,
    },
    "optimizer": {
        "learning_rate": 1e-3,
    },
    "training": {
        "batch_size": 32,
        "epochs": 25,
        "num_workers": 4
    },
    "loss": {
        "name": 'cross_entropy',
    },
}
