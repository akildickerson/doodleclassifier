import torch
from config import CONFIG
from models import CNN
from dataset import DoodleDataset
from torch.utils.data import DataLoader, random_split
from train import predict, train_one_epoch


def init_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_dataloaders(csv_file, root_dir) -> [DataLoader, DataLoader, DataLoader]:
    data = DoodleDataset(csv_file, root_dir)
    size = len(data)
    train_size = int(0.8 * size)
    val_size = int(0.1 * size)
    test_size = size - train_size - val_size

    train, val, test = random_split(
        data,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main() -> None:
    csv_file = (
        "/Users/akildickerson/Projects/DoodleClassifier/data/processed/doodlelabels.csv"
    )
    root_dir = "/Users/akildickerson/Projects/DoodleClassifier/data"
    train_loader, val_loader, test_loader = init_dataloaders(csv_file, root_dir)

    model = CNN()
    device = init_device()
    model.to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=CONFIG["optimizer"]["learning_rate"]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(CONFIG["training"]["epochs"]):
        res = train_one_epoch(model, train_loader, val_loader, optim, loss_fn, device)
        print(
            f"train loss: {res["train loss"]:.2f}\ttrain accuracy: {res["train accuracy"]:.2f}\tvalidation loss: {res["val loss"]:.2f}\tvalidation accuracy: {res["val accuracy"]:.2f}"
        )
    accuracy = predict(model, test_loader, device)
    print(f"-------- TEST ACCURACY --------\n\t\t{accuracy:.2f}")


if __name__ == "__main__":
    main()
