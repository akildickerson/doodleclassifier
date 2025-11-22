CONFIG = {
    "model": {
        "in_channels": 1,
        "conv_channels": [16, 32],
        "hidden_dims": [288, 128],
        "num_classes": 14,
    },
    "optimizer": {
        "learning_rate": 1e-3,
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "num_workers": 4
    },
    "loss": {
        "name": 'cross_entropy',
    },
}
