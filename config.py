CONFIG = {
    "model": {
        "in_channels": 1,
        "conv_channels": [8, 16],
        "hidden_dims": [400, 256],
        "num_classes": 14,
    },
    "optimizer": {
        "learning_rate": 1e-3,
    },
    "training": {
        "batch_size": 64,
        "epochs": 25,
        "num_workers": 4
    },
    "loss": {
        "name": 'cross_entropy',
    },
}
