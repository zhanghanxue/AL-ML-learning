import torch
import torch.nn as nn
from torch.optim import Adam
import time
import argparse
import os
from dataset import get_cifar10_dataloaders, get_oxford_pets_dataloaders
from model import SmallCNN, OxfordPetsCNN


class Trainer:
    def __init__(self, model, train_loader, val_loader, classes, device, lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = classes
        self.device = device
        
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_acc = 0.0
        self.best_class_stats = None
        self.start_epoch = 0

    def load_checkpoint(self, checkpoint_path="best_model.pth"):
        """Load model checkpoint if it exists"""
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_acc = checkpoint['best_acc']
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_class_stats = checkpoint.get('class_stats', None)
                
                print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
                print(f"  Best accuracy so far: {self.best_acc:.4f}")
                return True
            except Exception as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("  Starting training from scratch...")
        return False

    def save_checkpoint(self, epoch, class_stats=None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'class_stats': class_stats
        }
        torch.save(checkpoint, "best_model.pth")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        class_correct = {classname: 0 for classname in self.classes}
        class_total = {classname: 0 for classname in self.classes}
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update per-class statistics
                for label, pred in zip(labels, predicted):
                    class_idx = label.item()
                    if label == pred:
                        class_correct[self.classes[class_idx]] += 1
                    class_total[self.classes[class_idx]] += 1
        
        accuracy = correct / total
        val_loss /= len(self.val_loader)
        
        return val_loss, accuracy, class_correct, class_total

    def print_class_accuracy(self, class_correct, class_total):
        print("\nPer-class accuracy:")
        for classname in self.classes:
            correct_count = class_correct[classname]
            total_count = class_total[classname]
            accuracy = 100 * correct_count / total_count if total_count > 0 else 0
            print(f'  {classname:15s}: {accuracy:5.1f}% ({correct_count:4d}/{total_count:4d})')

    def train(self, epochs=3, resume=True):
        # Try to load checkpoint if resume is True
        checkpoint_loaded = False
        if resume:
            checkpoint_loaded = self.load_checkpoint()
        
        if checkpoint_loaded:
            print(f"Resuming training from epoch {self.start_epoch}")
        else:
            print("Starting training from scratch")
            self.start_epoch = 0
            self.best_acc = 0.0
        
        print(f"Training for {epochs} epochs on {self.device} (epochs {self.start_epoch} to {self.start_epoch + epochs - 1})")
        print(f"Classes: {', '.join(self.classes)}")
        print("-" * 60)
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss, accuracy, class_correct, class_total = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{self.start_epoch + epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"acc={accuracy:.4f}, "
                  f"time={epoch_time:.1f}s")
            
            # Save best model
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_class_stats = (class_correct.copy(), class_total.copy())
                self.save_checkpoint(epoch, self.best_class_stats)
                print("  ↳ New best model saved!")
        
        # Final results
        self.print_final_results()

    def print_final_results(self):
        if self.best_class_stats is None:
            # If no best stats available, get current model stats
            _, accuracy, class_correct, class_total = self.validate()
            self.best_class_stats = (class_correct, class_total)
            self.best_acc = accuracy
            
        class_correct, class_total = self.best_class_stats
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS (Best Model)")
        print("=" * 60)
        print(f"Best Overall Accuracy: {self.best_acc:.4f} ({self.best_acc*100:.2f}%)")
        print("\nDetailed Per-Class Accuracy:")
        
        for classname in self.classes:
            correct = class_correct[classname]
            total = class_total[classname]
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"  {classname:15s}: {accuracy:5.1f}% ({correct:4d}/{total:4d})")


def main():
    parser = argparse.ArgumentParser(description='Train on various datasets')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'oxford_pets'], 
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--resume', action='store_true', default=True, 
                       help='Resume training from checkpoint if available')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Choose dataset
    if args.dataset == 'cifar10':
        train_loader, val_loader, classes = get_cifar10_dataloaders(batch_size=args.bs)
        model = SmallCNN(num_classes=len(classes))
        
    elif args.dataset == 'oxford_pets':
        train_loader, val_loader, classes = get_oxford_pets_dataloaders(batch_size=args.bs)
        model = OxfordPetsCNN(num_classes=len(classes))  # Now outputs 37 classes instead of 10
    
    print(f"Using {args.dataset} with {len(classes)} classes")
    print(f"Model: {model.__class__.__name__}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, classes, device, lr=args.lr)
    trainer.train(epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()