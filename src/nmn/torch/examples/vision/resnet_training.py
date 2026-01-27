# main.py
#
# To run this script, you'll need to install the following packages:
# pip install torch torchvision torchaudio
# pip install datasets nmn-pytorch # For the original YAT models and TinyImageNet
# pip install wandb matplotlib seaborn scikit-learn
#
# Example usage:
# python main.py --model standard --dataset CIFAR10 --num-blocks 2 2 2 2 --epochs 50 --lr 0.003 --use-wandb
# python main.py --model yat --dataset TinyImageNet --num-blocks 2 2 2 2 --epochs 50 --lr 0.003 --use-wandb --image-size 64
# python main.py --model standard --hf-dataset food101 --num-blocks 2 2 2 2 --epochs 10 --lr 0.003 --use-wandb --image-size 224
#
# To login to W&B, run `wandb login` in your terminal.

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets as torchvision_datasets, transforms
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Suppress warning from the datasets library
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='datasets.builder')


# ---------- YAT convolution imports ----------
# Ensure you have the nmn-pytorch library installed: pip install nmn-pytorch
try:
    from nmn.torch.conv import YatConv2d
    from nmn.torch.nmn import YatNMN
except ImportError:
    print("Please install nmn-pytorch: pip install nmn-pytorch")
    exit()


# ---------- Dataset Handling ----------
class HuggingFaceDataset(torch.utils.data.IterableDataset):
    """Generic wrapper for streaming image classification datasets from Hugging Face Hub."""
    def __init__(self, dataset_name, split, transform=None, shuffle_buffer_size=50000):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        if split == 'train':
            # Shuffle the training data with a large buffer for better randomness
            self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer_size, seed=int(time.time()))
        self.transform = transform
        self.info = load_dataset(dataset_name, split=split).info

    def __iter__(self):
        for sample in self.dataset:
            # Common column names for image and label in HF datasets
            image_key = 'image' if 'image' in sample else 'img'
            label_key = 'label' if 'label' in sample else 'labels'
            
            if image_key not in sample or label_key not in sample:
                raise KeyError(f"Dataset sample missing required key. Found: {list(sample.keys())}")

            img_data = sample[image_key]
            if not isinstance(img_data, Image.Image):
                 img = img_data.convert("RGB")
            else:
                 img = img_data
            
            if self.transform:
                img = self.transform(img)
            
            yield img, sample[label_key]

def get_data_loaders(dataset_name, hf_dataset, batch_size, image_size, data_dir='./data'):
    """Creates train and validation data loaders for specified dataset."""
    
    # Define more powerful data augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # More advanced augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats as a general default
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32), # Resize to a slightly larger size
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if hf_dataset:
        print(f"Using Hugging Face dataset: {hf_dataset}")
        train_dataset = HuggingFaceDataset(hf_dataset, "train", train_transform)
        # Try common validation split names
        try:
            val_dataset = HuggingFaceDataset(hf_dataset, "validation", val_transform)
        except Exception:
            try:
                val_dataset = HuggingFaceDataset(hf_dataset, "test", val_transform)
            except Exception as e:
                raise ValueError(f"Could not load 'validation' or 'test' split for {hf_dataset}: {e}")

        num_classes = train_dataset.info.features['label'].num_classes
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        return train_loader, val_loader, num_classes

    # Adjust transforms for specific well-known datasets if needed
    if 'CIFAR' in dataset_name:
        # Use specific normalization stats for CIFAR
        if dataset_name == 'CIFAR10':
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        else: # CIFAR100
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomErasing(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    
    if dataset_name == 'CIFAR10':
        train_dataset = torchvision_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = torchvision_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        train_dataset = torchvision_datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = torchvision_datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
        num_classes = 100
    elif dataset_name == 'TinyImageNet':
         train_dataset = HuggingFaceDataset("zh-plus/tiny-imagenet", "train", train_transform)
         val_dataset = HuggingFaceDataset("zh-plus/tiny-imagenet", "valid", val_transform)
         num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use --hf-dataset for Hugging Face datasets.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes


# ---------- Models (Refactored with ResNet-style blocks) ----------

class BasicStandardBlock(nn.Module):
    """A basic residual block for the StandardConvNet, inspired by ResNet."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicStandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicYATBlock(nn.Module):
    """A basic residual block for the YATConvNet, inspired by ResNet."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
        super(BasicYATBlock, self).__init__()
        self.yat_conv = YatConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                  use_alpha=use_alpha, use_dropconnect=use_dropconnect,
                                  drop_rate=drop_rate, bias=False, epsilon=0.007)
        self.lin_conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.yat_conv(x, deterministic=not self.training)
        out = self.lin_conv(out)
        out += identity
        return out

class StandardConvNet(nn.Module):
    """A standard CNN with a ResNet-like architecture."""
    def __init__(self, block, num_blocks, num_classes=10):
        super(StandardConvNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class YATConvNet(nn.Module):
    """A YAT-based CNN with a ResNet-like architecture."""
    def __init__(self, block, num_blocks, num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
        super(YATConvNet, self).__init__()
        self.in_planes = 64
        self.use_alpha = use_alpha
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.proj = nn.Linear(512 * block.expansion, 256, bias=False)
        self.fc_yat = YatNMN(256, num_classes, epsilon=0.007, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s,
                                use_alpha=self.use_alpha,
                                use_dropconnect=self.use_dropconnect,
                                drop_rate=self.drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.proj(out)
        out = self.fc_yat(out)
        return out


# ---------- Analysis and Visualization ----------
def log_weight_histograms(model, epoch, use_wandb=False):
    """Logs histograms of model weights to W&B."""
    if not use_wandb:
        return

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    histograms = {}
    for name, param in actual_model.named_parameters():
        if param.requires_grad:
            histograms[f'weights/{name}'] = wandb.Histogram(param.data.cpu())
    wandb.log(histograms, step=epoch)

def plot_confusion_matrix(all_preds, all_targets, class_names, use_wandb=False):
    """Plots and logs a confusion matrix."""
    if not use_wandb:
        return

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


# ---------- Training and Validation ----------
def train_epoch(model, trainloader, optimizer, criterion, device, use_wandb, global_step):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        total_correct += predicted.eq(targets).sum().item()

        if batch_idx > 0 and batch_idx % 100 == 0:
            batch_loss = running_loss / 100
            batch_acc = 100. * total_correct / total_samples
            print(f'  Batch {batch_idx}/{len(trainloader)} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%')
            if use_wandb:
                wandb.log({
                    'train/batch_loss': batch_loss,
                    'train/batch_acc': batch_acc,
                }, step=global_step)
            running_loss = 0.0
        
        global_step += 1

    epoch_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
    epoch_acc = 100. * total_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc, global_step

def validate(model, valloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = test_loss / len(valloader) if len(valloader) > 0 else 0.0
    acc = 100. * correct / total if total > 0 else 0.0
    return avg_loss, acc, all_preds, all_targets


# ---------- Main Execution ----------
def main():
    parser = argparse.ArgumentParser(description='YAT/Standard ConvNet Training')
    # Model and Data
    parser.add_argument('--model', type=str, choices=['yat', 'standard'], default='yat', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10', help='Built-in dataset to use')
    parser.add_argument('--hf-dataset', type=str, default=None, help='Hugging Face dataset name (e.g., "food101", "beans")')
    parser.add_argument('--num-blocks', type=int, nargs='+', default=[2, 2, 2, 2], help='Number of blocks in each of the 4 ResNet stages')
    parser.add_argument('--image-size', type=int, default=32, help='Input image size')
    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    # YAT Specific
    parser.add_argument('--no-alpha', action='store_false', dest='use_alpha', help='Disable alpha in YATConv')
    parser.add_argument('--use-dropconnect', action='store_true', default=False, help='Use DropConnect in YATConv')
    parser.add_argument('--drop-rate', type=float, default=0.1, help='DropConnect rate')
    # System
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for datasets')
    # W&B
    parser.add_argument('--use-wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='yat-experiments', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username or team)')

    args = parser.parse_args()
    
    if args.hf_dataset and args.dataset != 'CIFAR10':
        print(f"Warning: --hf-dataset ('{args.hf_dataset}') is provided, ignoring --dataset ('{args.dataset}').")
        args.dataset = None


    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')

    # --- W&B Setup ---
    if args.use_wandb:
        run_name = f"{args.model}-{args.hf_dataset or args.dataset}-sz{args.image_size}-lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=run_name
        )

    # --- Data Loading ---
    print(f'Loading data...')
    train_loader, val_loader, num_classes = get_data_loaders(
        dataset_name=args.dataset,
        hf_dataset=args.hf_dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir
    )
    print(f'Dataset loaded. Num classes: {num_classes}')

    # --- Model Initialization ---
    if len(args.num_blocks) != 4:
        raise ValueError("The --num-blocks argument must have 4 integers for the 4 stages.")

    if args.model == 'yat':
        print(f'Creating YAT ResNet with blocks: {args.num_blocks}')
        model = YATConvNet(BasicYATBlock, args.num_blocks, num_classes=num_classes, use_alpha=args.use_alpha,
                           use_dropconnect=args.use_dropconnect, drop_rate=args.drop_rate)
    else:
        print(f'Creating Standard ResNet with blocks: {args.num_blocks}')
        model = StandardConvNet(BasicStandardBlock, args.num_blocks, num_classes=num_classes)

    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    if args.use_wandb:
        wandb.watch(model, log='all', log_freq=100)

    # --- Optimizer, Scheduler, Criterion ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    best_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f'\nEpoch {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')

        train_loss, train_acc, global_step = train_epoch(model, train_loader, optimizer, criterion, device, args.use_wandb, global_step)
        val_loss, val_acc, all_preds, all_targets = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch Summary: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}')

        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_acc': train_acc,
                'train/epoch_loss': train_loss,
                'val/acc': val_acc,
                'val/loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            }, step=global_step)
            log_weight_histograms(model, epoch, args.use_wandb)

        if val_acc > best_acc:
            best_acc = val_acc
            print(f'  -> New best validation accuracy: {best_acc:.2f}%. Saving model...')
            model_path = f'best_{args.model}_{args.hf_dataset or args.dataset}.pth'
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, model_path)
            if args.use_wandb:
                wandb.save(model_path)

        print(f'Epoch time: {time.time()-start_time:.1f}s | Best Val Acc: {best_acc:.2f}%')

    # --- Final Evaluation and Logging ---
    print('\nTraining completed!')
    if args.use_wandb:
        model_path = f'best_{args.model}_{args.hf_dataset or args.dataset}.pth'
        state_dict = torch.load(model_path, map_location=device)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(state_dict)

        _, _, all_preds, all_targets = validate(model, val_loader, criterion, device)
        
        # Get class names for confusion matrix
        if hasattr(train_loader.dataset, 'classes'):
            class_names = train_loader.dataset.classes
        elif hasattr(train_loader.dataset, 'info'):
             class_names = train_loader.dataset.info.features['label'].names
        else:
            class_names = [str(i) for i in range(num_classes)]

        plot_confusion_matrix(all_preds, all_targets, class_names, args.use_wandb)

        wandb.finish()

if __name__ == '__main__':
    main()