"""
CIFAR-10 MLP-Mixer with YAT (ⵟ-product) Implementation
Architecture respects Representer Theorem: YAT layer → Linear projection

CRITICAL DESIGN NOTE:
Following the representer theorem (Schölkopf et al., 2001), composing multiple 
YAT-layers without intervening linear projections would create a deep kernel 
that loses the representational guarantee. Therefore, we MUST pair each YAT 
layer with a subsequent linear projection.

Run with:
    python cifar10_yat_mixer.py --architecture yat-mixer --epochs 50
    python cifar10_yat_mixer.py --architecture standard-mixer --epochs 50
    python cifar10_yat_mixer.py --architecture quadratic-mixer --epochs 50
    python cifar10_yat_mixer.py --architecture no-norm-mixer --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class YATLayer(nn.Module):
    """
    YAT (ⵟ-product) Layer: K_ⵟ(w,x) = (w^T x)^2 / (||w-x||^2 + ε)
    
    Properties:
    - Mercer kernel (Theorem 1)
    - Self-regulating (Proposition 1): output bounded without normalization
    - Lipschitz continuous (Proposition 3)
    - Infinitely differentiable (Lemma 1)
    
    CRITICAL: Must be followed by linear projection to preserve kernel properties.
    """
    def __init__(self, in_features, out_features, epsilon=0.1, 
                 init_mode='normal', init_rectifier='none',
                 drop_connect_rate=0.0, spherical=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon
        self.drop_connect_rate = drop_connect_rate
        self.spherical = spherical
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialization logic
        if init_mode == 'normal':
            # Kaiming Normal-like std for linear layers is gain/sqrt(fan_in)
            # Default PyTorch Linear uses 1/sqrt(fan_in) for both weights and bias init
            std = 1.0 / (in_features ** 0.5)
            nn.init.normal_(self.weight, mean=0.0, std=std)
        elif init_mode == 'uniform':
            bound = 1.0 / (in_features ** 0.5)
            nn.init.uniform_(self.weight, -bound, bound)
            
        # Apply rectification if requested
        if init_rectifier == 'abs':
            with torch.no_grad():
                self.weight.abs_()
        elif init_rectifier == 'relu':
            with torch.no_grad():
                self.weight.relu_()
        
        # Adaptive scaling factor: s = sqrt(n / log(1+n))^α
        self.alpha = nn.Parameter(torch.ones(1))
        scale_base_val = out_features / torch.log(torch.tensor(out_features + 1.0)).item()
        self.register_buffer('scale_base', torch.tensor(scale_base_val))
    
    def forward(self, x):
        """
        Forward pass computing YAT kernel.
        
        Args:
            x: (batch_size, ..., in_features) - can handle arbitrary batch dimensions
        
        Returns:
            yat_output: (batch_size, ..., out_features)
        """
        # Store original shape for reshaping
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        
        # Flatten batch dimensions: (..., in_features) -> (N, in_features)
        x_flat = x.reshape(-1, self.in_features)
        
        weight = self.weight
        if self.training and self.drop_connect_rate > 0:
            weight = F.dropout(weight, p=self.drop_connect_rate)

        if self.spherical:
            # Normalize x and w
            x_flat = F.normalize(x_flat, p=2, dim=1)
            weight = F.normalize(weight, p=2, dim=1)

        # Compute dot product with bias: w^T x + b
        dot_product = F.linear(x_flat, weight, self.bias)  # (N, out_features)
        
        if self.spherical:
             # Spherical YAT: inputs and kernel are normalized
            # distances = ||x||² + ||W||² - 2(x · W) = 1 + 1 - 2(x · W) = 2 - 2(x · W)
            # Bias removed for distance calc as per original implementation logic
            
            # Recompute dot without bias for distance to be consistent
            dot_for_dist = F.linear(x_flat, weight)
            distance_sq = 2 - 2 * dot_for_dist
            
        else:
            # Efficient distance computation using algebraic identity:
            # ||w - x||^2 = ||w||^2 + ||x||^2 - 2*w^T*x
            x_norm_sq = (x_flat ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            w_norm_sq = (weight ** 2).sum(dim=1).unsqueeze(0)  # (1, out_features)
            
            # Distance squared (without bias for distance calculation)
            dot_product_no_bias = F.linear(x_flat, weight)  # (N, out_features)
            distance_sq = w_norm_sq + x_norm_sq - 2 * dot_product_no_bias
            
        denominator = distance_sq + self.epsilon
        
        # YAT kernel: (w^T x + b)^2 / (||w-x||^2 + ε)
        numerator = dot_product ** 2
        yat_output = numerator / denominator
        
        # Adaptive scaling: s = sqrt(n / log(1+n))^α
        scale = torch.sqrt(self.scale_base) ** self.alpha
        yat_output = scale * yat_output
        
        # Reshape back to original batch dimensions
        output_shape = batch_dims + (self.out_features,)
        yat_output = yat_output.reshape(output_shape)
        
        return yat_output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class QuadraticLayer(nn.Module):
    """Quadratic neuron baseline: (Wx)^2"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        linear_out = F.linear(x, self.weight, self.bias)
        return linear_out ** 2


class PolynomialLayer(nn.Module):
    """Polynomial kernel baseline: (w^T x)^2"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))
        
    def forward(self, x):
        dot_product = F.linear(x, self.weight)
        return dot_product ** 2


class MixerBlock(nn.Module):
    """
    MLP-Mixer block with YAT replacing standard MLPs.
    
    Architecture (YAT mode):
        Token-mixing:  YAT(num_patches → tokens_mlp_dim) → Linear(tokens_mlp_dim → num_patches)
        Channel-mixing: YAT(hidden_dim → channels_mlp_dim) → Linear(channels_mlp_dim → hidden_dim)
    
    Following representer theorem: Each YAT layer is paired with a linear projection.
    Self-regulation property eliminates need for LayerNorm.
    """
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, 
                 architecture='yat-mixer', epsilon=0.1, dropout=0.0,
                 init_mode='normal', init_rectifier='none', drop_path=0.0,
                 drop_connect_rate=0.0, spherical=False):
        super().__init__()
        self.architecture = architecture
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Token-mixing MLP
        if architecture == 'yat-mixer':
            # YAT → Linear (respects representer theorem)
            self.token_mix_1 = YATLayer(num_patches, tokens_mlp_dim, epsilon,
                                      init_mode=init_mode, init_rectifier=init_rectifier,
                                      drop_connect_rate=drop_connect_rate, spherical=spherical)
            self.token_mix_2 = nn.Linear(tokens_mlp_dim, num_patches)
            
        elif architecture == 'quadratic-mixer':
            self.token_mix_1 = QuadraticLayer(num_patches, tokens_mlp_dim)
            self.token_mix_2 = nn.Linear(tokens_mlp_dim, num_patches)
            
        elif architecture == 'polynomial-mixer':
            self.token_mix_1 = PolynomialLayer(num_patches, tokens_mlp_dim)
            self.token_mix_2 = nn.Linear(tokens_mlp_dim, num_patches)
            
        elif architecture == 'no-norm-mixer':
            # Standard Linear + GELU but no normalization
            self.token_mix_1 = nn.Linear(num_patches, tokens_mlp_dim)
            self.token_mix_2 = nn.Linear(tokens_mlp_dim, num_patches)
            
        else:  # standard-mixer
            self.token_mix_1 = nn.Linear(num_patches, tokens_mlp_dim)
            self.token_mix_2 = nn.Linear(tokens_mlp_dim, num_patches)
        
        # Channel-mixing MLP (same pattern)
        if architecture == 'yat-mixer':
            self.channel_mix_1 = YATLayer(hidden_dim, channels_mlp_dim, epsilon,
                                        init_mode=init_mode, init_rectifier=init_rectifier,
                                        drop_connect_rate=drop_connect_rate, spherical=spherical)
            self.channel_mix_2 = nn.Linear(channels_mlp_dim, hidden_dim)
            
        elif architecture == 'quadratic-mixer':
            self.channel_mix_1 = QuadraticLayer(hidden_dim, channels_mlp_dim)
            self.channel_mix_2 = nn.Linear(channels_mlp_dim, hidden_dim)
            
        elif architecture == 'polynomial-mixer':
            self.channel_mix_1 = PolynomialLayer(hidden_dim, channels_mlp_dim)
            self.channel_mix_2 = nn.Linear(channels_mlp_dim, hidden_dim)
            
        elif architecture == 'no-norm-mixer':
            self.channel_mix_1 = nn.Linear(hidden_dim, channels_mlp_dim)
            self.channel_mix_2 = nn.Linear(channels_mlp_dim, hidden_dim)
            
        else:  # standard-mixer
            self.channel_mix_1 = nn.Linear(hidden_dim, channels_mlp_dim)
            self.channel_mix_2 = nn.Linear(channels_mlp_dim, hidden_dim)
        
        # LayerNorm only for standard-mixer
        self.use_norm = (architecture == 'standard-mixer')
        if self.use_norm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, hidden_dim)
        """
        # Token-mixing
        if self.use_norm:
            shortcut = x
            x = self.norm1(x)
        else:
            shortcut = x
            
        # Transpose for token-mixing: (batch, hidden_dim, num_patches)
        x = x.transpose(1, 2)
        
        # Apply token-mixing MLP
        if self.architecture in ['no-norm-mixer', 'standard-mixer']:
            x = F.gelu(self.token_mix_1(x))
        else:
            x = self.token_mix_1(x)  # YAT/Quadratic/Polynomial already non-linear
            
        x = self.dropout(self.token_mix_2(x))
        
        # Transpose back: (batch, num_patches, hidden_dim)
        x = x.transpose(1, 2)
        x = shortcut + self.drop_path(x)
        
        # Channel-mixing
        if self.use_norm:
            shortcut = x
            x = self.norm2(x)
        else:
            shortcut = x
            
        # Apply channel-mixing MLP
        if self.architecture in ['no-norm-mixer', 'standard-mixer']:
            x = F.gelu(self.channel_mix_1(x))
        else:
            x = self.channel_mix_1(x)
            
        x = self.dropout(self.channel_mix_2(x))
        x = shortcut + self.drop_path(x)
        
        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer for CIFAR-10 with YAT layers.
    
    Architecture:
        1. Patch embedding: Conv2d to split image into patches
        2. N × MixerBlock (token + channel mixing)
        3. Global average pooling
        4. Linear classifier
    """
    def __init__(self, num_classes=10, patch_size=4, hidden_dim=256, num_blocks=8,
                 tokens_mlp_dim=256, channels_mlp_dim=512, architecture='yat-mixer',
                 epsilon=0.1, dropout=0.0, init_mode='normal', init_rectifier='none',
                 drop_path_rate=0.0, drop_connect_rate=0.0, spherical=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # CIFAR-10: 32x32 → (32/patch_size)^2 patches
        num_patches = (32 // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        # Mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim,
                      architecture, epsilon, dropout, init_mode, init_rectifier,
                      drop_path=dpr[i], drop_connect_rate=drop_connect_rate, spherical=spherical)
            for i in range(num_blocks)
        ])
        
        # Classifier
        self.use_norm = (architecture == 'standard-mixer')
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.head = nn.Linear(hidden_dim, num_classes)
        # Enforce orthogonal initialization for the output layer
        nn.init.orthogonal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 32, 32)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Patch embedding: (batch, 3, 32, 32) → (batch, hidden_dim, H, W)
        x = self.patch_embed(x)
        
        # Flatten patches: (batch, hidden_dim, H, W) → (batch, num_patches, hidden_dim)
        batch_size, hidden_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Apply mixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling: (batch, num_patches, hidden_dim) → (batch, hidden_dim)
        x = x.mean(dim=1)
        
        # Optional normalization
        if self.use_norm:
            x = self.norm(x)
        
        # Classifier
        x = self.head(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, optimizer, criterion, device, use_mixup=False, mixup_alpha=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    epoch_losses = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, use_cuda=(device=='cuda'))
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        epoch_losses.append(loss_val)
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        if use_mixup:
            # Approx accuracy (using the dominant label)
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total, epoch_losses


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 MLP-Mixer with YAT')
    parser.add_argument('--architecture', type=str, default='yat-mixer',
                       choices=['yat-mixer', 'standard-mixer', 'quadratic-mixer', 
                               'polynomial-mixer', 'no-norm-mixer'],
                       help='Architecture variant')
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-blocks', type=int, default=8, help='Number of mixer blocks')
    parser.add_argument('--tokens-mlp-dim', type=int, default=256, help='Token MLP dimension')
    parser.add_argument('--channels-mlp-dim', type=int, default=512, help='Channel MLP dimension')
    parser.add_argument('--epsilon', type=float, default=0.001, help='YAT epsilon parameter')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='./results', help='Save directory')
    # New initialization arguments
    parser.add_argument('--init-mode', type=str, default='normal', choices=['normal', 'uniform'],
                       help='Initialization mode for YAT layers')
    parser.add_argument('--init-rectifier', type=str, default='none', choices=['none', 'abs', 'relu'],
                       help='Initialization rectifier for YAT layers')
    
    # Generalization improvements
    parser.add_argument('--rand-augment', action='store_true', help='Use RandAugment')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp')
    parser.add_argument('--mixup-alpha', type=float, default=1.0, help='MixUp alpha')
    parser.add_argument('--drop-path', type=float, default=0.0, help='Drop Path (Stochastic Depth) rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--drop-connect', type=float, default=0.0, help='Drop Connect rate for YAT weights')
    parser.add_argument('--spherical', action='store_true', help='Use Spherical YAT')
    parser.add_argument('--escape-plateau-patience', type=int, default=0, help='Patience for switching to SGD (0 to disable)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data augmentation and normalization
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    
    if args.rand_augment:
        # RandAugment for CIFAR
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            transform_list.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
            print("Using AutoAugment (CIFAR10 Policy)")
        except ImportError:
            print("WARNING: torchvision too old for AutoAugment, skipping.")
            
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_train = transforms.Compose(transform_list)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                                     transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = MLPMixer(
        num_classes=10,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        tokens_mlp_dim=args.tokens_mlp_dim,
        channels_mlp_dim=args.channels_mlp_dim,
        architecture=args.architecture,
        epsilon=args.epsilon,
        dropout=args.dropout,
        init_mode=args.init_mode,
        init_rectifier=args.init_rectifier,
        drop_path_rate=args.drop_path,
        drop_connect_rate=args.drop_connect,
        spherical=args.spherical
    ).to(args.device)
    
    num_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Architecture: {args.architecture}")
    print(f"Init Mode: {args.init_mode}, {args.init_rectifier}")
    print(f"Augmentations: RandAugment={args.rand_augment}, MixUp={args.mixup}")
    print(f"Regularization: DropPath={args.drop_path}, LabelSmooth={args.label_smoothing}, Warmup={args.warmup_epochs}")
    print(f"YAT Options: DropConnect={args.drop_connect}, Spherical={args.spherical}")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"{'='*60}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                  weight_decay=args.weight_decay)
                                  
    # Scheduler with Warmup
    if args.warmup_epochs > 0:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training loop
    best_acc = 0
    best_val_loss = float('inf')
    plateau_counter = 0
    switched_to_sgd = False
    
    history = []
    global_iteration_losses = []  # Store all iteration losses
    
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<8}")
    print('-' * 72)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc, epoch_losses = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            use_mixup=args.mixup, mixup_alpha=args.mixup_alpha
        )
        global_iteration_losses.extend(epoch_losses)
        
        val_loss, val_acc = validate(model, test_loader, criterion, args.device)
        
        # Escape Plateau Logic
        if args.escape_plateau_patience > 0 and not switched_to_sgd:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            if plateau_counter >= args.escape_plateau_patience:
                print(f"\n[Escape Plateau] Validation loss plateaued for {plateau_counter} epochs.")
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Escape Plateau] Switching to SGD with LR={current_lr}")
                
                # Switch to SGD
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9)
                
                # Re-create scheduler for the new optimizer, attempting to preserve schedule
                # Assuming CosineAnnealingLR for simplicity as it's the main usage
                if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                     # Still in warmup (unlikely to plateau, but handle it)
                     # Re-create the composite scheduler
                     main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
                     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
                     scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs], last_epoch=epoch)
                else:
                    # Past warmup or no warmup
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=epoch)
                
                switched_to_sgd = True
                plateau_counter = 0 # Reset counter

        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Print progress
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f} {epoch_time:<8.2f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, save_dir / f'{args.architecture}_best.pth')
    
    print(f"\n{'='*60}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}\n")

    # Plot iteration losses
    plt.figure(figsize=(10, 6))
    plt.plot(global_iteration_losses, label='Iteration Loss')
    # Add a moving average for cleaner visualization
    window_size = 50
    if len(global_iteration_losses) > window_size:
        moving_avg = np.convolve(global_iteration_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(global_iteration_losses)), moving_avg, label=f'Moving Avg ({window_size})', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Iteration ({args.architecture})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'{args.architecture}_loss_history.png')
    plt.close()
    
    # Save training history
    with open(save_dir / f'{args.architecture}_history.json', 'w') as f:
        json.dump({
            'args': vars(args),
            'num_parameters': num_params,
            'best_val_acc': best_acc,
            'history': history,
            'iteration_losses': global_iteration_losses
        }, f, indent=2)
    
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()