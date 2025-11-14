import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_dataloaders(batch_size=64, img_size=32):
    """
    Creates DataLoaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch. Defaults to 64.
        img_size (int): Size to resize images to (img_size x img_size). 
    
    Returns:
        train_loader, valid_loader, classes: Training and validation DataLoaders and the list of class names.
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform)
    
    valid_dataset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, valid_loader, train_dataset.classes

def get_oxford_pets_dataloaders(batch_size=64, img_size=224, validation_split=0.2):
    """
    Creates DataLoaders for the Oxford-IIIT Pet dataset.
    
    Args:
        batch_size (int): Batch size for training and validation.
        img_size (int): Height and width to which images are resized.
        validation_split (float): Proportion of data to use for validation (e.g., 0.2 for 20%).
    
    Returns:
        train_loader, val_loader, classes: Training and validation DataLoaders and the list of class names.
    """
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),  # Data augmentation for training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    
    full_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",  # Use 'category' for 37-class breed classification:cite[1]
        download=True,
        transform=data_transforms['train']
    )
    
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])
    
    valid_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, valid_loader, full_dataset.classes

if __name__ == "__main__":
    train_loader, valid_loader, classes = get_oxford_pets_dataloaders()
    xb, yb = next(iter(train_loader))
    print(f"Train batch: {xb.shape}, Labels: {yb.shape}, Classes: {classes[:5]}")