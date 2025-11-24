import torch
from config import CONFIG
from models import CNN
from dataset import DoodleDataset
from torch.utils.data import DataLoader, random_split
from train import predict, train_one_epoch


def init_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
    else:
        print("cpu")
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
    val_loader = DataLoader(val, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=64, shuffle=True, num_workers=4)

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

    best_acc = 0.0

    for epoch in range(CONFIG["training"]["epochs"]):
        res = train_one_epoch(model, train_loader, val_loader, optim, loss_fn, device)
        print(
            f"epoch {epoch+1:02d} | "
            f"train loss: {res["train loss"]:.2f}\t"
            f"train accuracy: {100 * res["train accuracy"]:.2f}%\t"
            f"validation loss: {res["val loss"]:.2f}\t"
            f"validation accuracy: {100 * res["val accuracy"]:.2f}%"
        )

        if res["val accuracy"] > best_acc:
            best_acc = res["val accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "best_val_acc": best_acc,
                },
                "best_model.pt",
            )
    
    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    accuracy = predict(model, test_loader, device)
    print(f"-------- TEST ACCURACY --------\n\t\t{accuracy:.2f}")


if __name__ == "__main__":
    main()
