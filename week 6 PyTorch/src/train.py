import torch, time
import torch.nn as nn
from torch.optim import Adam
from src.dataset import get_cifar10_dataloaders
from src.model import SmallCNN

def train(epochs=3, bs=128, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dl, valid_dl, classes = get_cifar10_dataloaders(batch_size=bs)
    model = SmallCNN(num_classes=len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss/len(train_dl)
        # validation
        model.eval()
        correct = total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                _, predicted = torch.max(predictions, 1)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(valid_dl)
        acc = correct/total
        print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={acc:.4f}, time={time.time()-t0:.1f}s")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
    return model

if __name__ == "__main__":
    trained_model = train(epochs=1, bs=64, lr=1e-3)