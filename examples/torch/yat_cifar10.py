"""
YAT Convolution CIFAR-10 Training Example

This script demonstrates how to use YAT (Yet Another Transformer) convolution layers
for image classification on CIFAR-10. YAT convolutions use a distance-based attention
mechanism that computes (dot_product)^2 / (distance_squared + epsilon) for each
patch-kernel pair.

Features demonstrated:
- YatConv2d layers with alpha scaling and DropConnect
- Comparison with standard Conv2d layers
- Training loop with validation
- Model performance evaluation

Usage:
    python yat_cifar10.py --model yat --epochs 20 --batch-size 128
    python yat_cifar10.py --model standard --epochs 20 --batch-size 128
"""

import argparse
import time
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add the parent directory to path to import YAT convolutions
from nmn.torch import YatConv2d
from nmn.torch import YatNMN


class YATConvNet(nn.Module):
    """CNN using YAT convolution layers for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10, use_alpha=True, use_dropconnect=True, drop_rate=0.1):
        super(YATConvNet, self).__init__()
        
        # First YAT conv block
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=32, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        
        
        
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Second YAT conv block
        self.conv3 = YatConv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            padding=1,
            use_alpha=True,
            use_dropconnect=False,
            bias=False
        )
            
        
        self.lin_conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Third YAT conv block
        self.conv5 = YatConv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            padding=1,
            use_alpha=True,
            use_dropconnect=False,
            bias=False
        )
        
        self.lin_conv5 = nn.Conv2d(
            in_channels=128, 
            out_channels=128, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes, bias=False)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv3(x, deterministic=not self.training)
        x = self.lin_conv3(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv5(x, deterministic=not self.training)
        x = self.lin_conv5(x)
        x = self.pool3(x)
        
        # Classifier
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x


class StandardConvNet(nn.Module):
    """Standard CNN using regular Conv2d layers for comparison."""
    
    def __init__(self, num_classes=10):
        super(StandardConvNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Classifier
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_data_loaders(batch_size=128, num_workers=2):
    """Create CIFAR-10 data loaders with standard augmentation."""
    
    # Data preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


def train_epoch(model, trainloader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx + 1:4d}: Loss: {running_loss / 100:.3f} | '
                  f'Acc: {100. * correct / total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total


def validate(model, testloader, criterion, device):
    """Validate the model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader)
    test_acc = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    return test_acc


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='YAT Convolution CIFAR-10 Training')
    parser.add_argument('--model', choices=['yat', 'standard'], default='yat',
                        help='Model type: yat or standard conv')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use-alpha', action='store_true', default=True,
                        help='Use alpha scaling in YAT convolutions')
    parser.add_argument('--use-dropconnect', action='store_true', default=False,
                        help='Use DropConnect in YAT convolutions')
    parser.add_argument('--drop-rate', type=float, default=0.1,
                        help='DropConnect rate for YAT layers (default: 0.1)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')
    
    # Data loaders
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_data_loaders(args.batch_size)
    
    # Model setup
    if args.model == 'yat':
        print('Creating YAT ConvNet...')
        model = YATConvNet(
            num_classes=10,
            use_alpha=args.use_alpha,
            use_dropconnect=args.use_dropconnect,
            drop_rate=args.drop_rate
        )
        print(f'YAT Conv settings: alpha={args.use_alpha}, '
              f'dropconnect={args.use_dropconnect}, drop_rate={args.drop_rate}')
    else:
        print('Creating Standard ConvNet...')
        model = StandardConvNet(num_classes=10)
    
    model = model.to(device)
    print(f'Model parameters: {count_parameters(model):,}')
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f'\nEpoch {epoch + 1}/{args.epochs}:')
        
        # Train
        train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        
        # Validate
        test_acc = validate(model, testloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': args,
            }, f'best_{args.model}_cifar10.pth')
        
        epoch_time = time.time() - start_time
        print(f'Epoch time: {epoch_time:.1f}s | Best test acc: {best_acc:.2f}%')
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    
    # Load and test the best model
    print('\nTesting best model...')
    checkpoint = torch.load(f'best_{args.model}_cifar10.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_acc = validate(model, testloader, criterion, device)
    
    print(f'Final test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
