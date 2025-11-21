"""Training & evaluation loops."""

import torch


def train_one_epoch(model, dataloader1, dataloader2, optimizer, loss_fn, device):
    model.train()
    train_loss, train_total, train_correct = 0.0, 0, 0
    for img, label in dataloader1:
        img, label = img.to(device), label.long().to(device)
        logits = model(img)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        train_loss += loss.item() * batch_size
        train_correct += (logits.argmax(dim=1) == label).sum().item()
        train_total += batch_size

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    model.eval()
    with torch.no_grad():
        val_loss, val_total, val_correct = 0.0, 0, 0
        for img, label in dataloader2:
            img, label = img.to(device), label.long().to(device)
            logits = model(img)
            loss = loss_fn(logits, label)

            pred = logits.argmax(dim=1)

            batch_size = label.size(0)
            val_loss += loss.item() * batch_size
            val_correct += (pred == label).sum().item()
            val_total += batch_size
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    return {
        "train loss": train_loss,
        "train accuracy": train_acc,
        "val loss": val_loss,
        "val accuracy": val_acc,
    }


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    correct, total = 0.0, 0.0
    for img, label in dataloader:
        img, label = img.to(device), label.long().to(device)
        logits = model(img)

        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    return correct / total  # accuracy = correct / total