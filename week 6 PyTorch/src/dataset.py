from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=64, img_size=32):
    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = datasets.CIFAR10(root='data', train=True, download=True, transform=tf)
    valid_ds = datasets.CIFAR10(root='data', train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, valid_dl, train_ds.classes

if __name__ == "__main__":
    train_dl, valid_dl, classes = get_cifar10_dataloaders()
    xb, yb = next(iter(train_dl))
    print(f"Train batch: {xb.shape}, Labels: {yb.shape}, Classes: {classes[:5]}")