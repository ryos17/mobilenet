import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import copy
from utils.mobilenet import MobileNetV1
from utils.femto_mobilenet import FemtoMobileNetV1
from utils.fused_layers import fuse_model
from sklearn.metrics import accuracy_score


class ImageDataset(Dataset):
    """Custom dataset for loading images from person and non_person directories."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_image_paths(data_dir):
    """Load all image paths from person and non_person directories."""
    person_dir = Path(data_dir) / 'person'
    non_person_dir = Path(data_dir) / 'non_person'
    
    # Get all image paths
    person_images = list(person_dir.glob('*.jpg'))
    non_person_images = list(non_person_dir.glob('*.jpg'))
    
    # Create labels: 0 for non_person, 1 for person
    image_paths = non_person_images + person_images
    labels = [0] * len(non_person_images) + [1] * len(person_images)
    
    print(f"Loaded {len(non_person_images)} non_person images")
    print(f"Loaded {len(person_images)} person images")
    print(f"Total: {len(image_paths)} images")
    
    return image_paths, labels


def split_dataset(image_paths, labels, train_ratio, val_ratio, test_ratio, seed=42):
    """Split dataset into train, validation, and test sets."""    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle indices
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    
    # Calculate split sizes
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create splits
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_transforms(is_train=True, rho=1.0, baseline_size=224):
    """Get ImageNet data transforms for training or validation.
    
    Args:
        is_train: Whether to use training transforms (with augmentation)
        rho: Resolution multiplier (default: 1.0 for baseline 224x224)
        baseline_size: Baseline input size (default: 224)
    """
    # Calculate input size based on resolution multiplier
    input_size = int(baseline_size * rho)
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Convert probabilities to log probabilities
        log_outputs = torch.log(outputs + 1e-8)  
        loss = criterion(log_outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)

            # Convert probabilities to log probabilities
            log_outputs = torch.log(outputs + 1e-8)  
            loss = criterion(log_outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNetV1 for Visual Wake Words')
    
    # Required arguments
    parser.add_argument('run_name', type=str,
                        help='Name of the run')
                        
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing person and non_person subdirectories (default: data)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of test data (default: 0.1)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='femto_mobilenet', choices=['mobilenet', 'femto_mobilenet'],
                        help='Type of model to use (default: femto_mobilenet)')
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Width multiplier for MobileNet (default: 1.0)')
    parser.add_argument('--baseline_size', type=int, default=224,
                        help='Baseline input image size (default: 224)')
    parser.add_argument('--rho', type=float, default=96/224,
                        help='Resolution multiplier for MobileNet (default: 96/224)')
    parser.add_argument('--ch_in', type=int, default=3,
                        help='Number of input channels (default: 3 for RGB)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes (default: 2 for binary classification)')
    parser.add_argument('--best_model_path', type=str, default='models/best_model.pth',
                        help='Path to save the best model (default: best_model.pth)')
    parser.add_argument('--fuse', action='store_true',
                        help='Fuse model (default: False)')
    parser.add_argument('--best_fused_model_path', type=str, default='models/best_fused_model.pth',
                        help='Path to save the best fused model (default: best_fused_model.pth)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--wandb_project', type=str, default='femto-mobilenet-vww',
                        help='Wandb project name (default: femto-mobilenet-vww)')
    parser.add_argument('--wandb_entity', type=str, default='deep-learning',
                        help='Wandb entity name (default: deep-learning)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None)')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}")
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Calculate input size based on resolution multiplier
    baseline_size = args.baseline_size
    input_size = int(baseline_size * args.rho)
    
    # Configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'run_name': args.run_name,
        'model_type': args.model_type,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'alpha': args.alpha,
        'rho': args.rho,
        'fuse': args.fuse,
        'input_size': input_size,
        'ch_in': args.ch_in,
        'n_classes': args.n_classes,
        'best_model_path': args.best_model_path,
        'best_fused_model_path': args.best_fused_model_path,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'device': device,
        'num_workers': args.num_workers,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'resume': args.resume
    }
    
    # Initialize wandb
    wandb.init(
        name=args.run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config
    )
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    print(f"Resolution multiplier (rho): {args.rho}")
    print(f"Input size: {input_size}x{input_size} (baseline: {baseline_size}x{baseline_size})")
    
    # Load image paths and labels
    image_paths, labels = load_image_paths(args.data_dir)
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        image_paths, labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create datasets with resolution multiplier
    train_dataset = ImageDataset(train_data[0], train_data[1], 
                                  transform=get_transforms(is_train=True, rho=args.rho, baseline_size=baseline_size))
    val_dataset = ImageDataset(val_data[0], val_data[1], 
                                transform=get_transforms(is_train=False, rho=args.rho, baseline_size=baseline_size))
    test_dataset = ImageDataset(test_data[0], test_data[1], 
                                 transform=get_transforms(is_train=False, rho=args.rho, baseline_size=baseline_size))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True)
    
    # Initialize model
    if args.model_type == 'mobilenet':
        model = MobileNetV1(ch_in=args.ch_in, n_classes=args.n_classes, alpha=args.alpha)
    elif args.model_type == 'femto_mobilenet':
        model = FemtoMobileNetV1(ch_in=args.ch_in, n_classes=args.n_classes, alpha=args.alpha)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model = model.to(device)

    # Initialize modules with a dummy forward pass to compute the input shape
    with torch.no_grad():
        dummy_input = torch.zeros(1, args.ch_in, input_size, input_size).to(device)
        _ = model(dummy_input)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Log parameters to wandb
    wandb.config.update({
        'total_params': total_params,
        'trainable_params': trainable_params
    })
    wandb.run.summary['total_params'] = total_params
    wandb.run.summary['trainable_params'] = trainable_params
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Resume optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Log model architecture to wandb
    wandb.watch(model, log='all', log_freq=100)
    
    # Training loop
    best_val_loss = float('inf')
    best_fused_val_loss = float('inf') if args.fuse else None
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if args.fuse:
            fused_model = copy.deepcopy(model)
            fused_model = fuse_model(fused_model)
            fused_val_loss, fused_val_acc = validate(fused_model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'fused_val_loss': fused_val_loss if args.fuse else None,
            'fused_val_acc': fused_val_acc if args.fuse else None,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"{'Train Loss:':<30} {train_loss:.4f}")
        print(f"{'Train Acc:':<30} {train_acc:.4f}")
        print(f"{'Val Loss:':<30} {val_loss:.4f}")
        print(f"{'Val Acc:':<30} {val_acc:.4f}")
        print(f"{'Fused Val Loss:':<30} {fused_val_loss:.4f}" if args.fuse else "")
        print(f"{'Fused Val Acc:':<30} {fused_val_acc:.4f}" if args.fuse else "")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, args.best_model_path)
            print(f"\n[Best Model Saved]  Val Loss: {best_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_val_acc'] = val_acc
        if args.fuse:
            if fused_val_loss < best_fused_val_loss:
                best_fused_val_loss = fused_val_loss
                torch.save({
                    'model_state_dict': fused_model.state_dict(),
                    'fused_val_loss': fused_val_loss,
                    'fused_val_acc': fused_val_acc
                }, args.best_fused_model_path)   
                print(f"[Best Fused Model Saved]  Fused Val Loss: {best_fused_val_loss:.4f}, Fused Val Acc: {fused_val_acc:.4f}")
                wandb.run.summary['best_fused_val_loss'] = best_fused_val_loss
                wandb.run.summary['best_fused_val_acc'] = fused_val_acc
    
    # Test on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")

    # Load best model
    checkpoint = torch.load(args.best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run test on best model
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    wandb.run.summary['test_loss'] = test_loss
    wandb.run.summary['test_acc'] = test_acc
    
    if args.fuse:
        # Fuse Conv+BN layers for faster inference
        print("\n" + "=" * 50)
        print("Fusing Conv+BN layers...")
        fused_model = copy.deepcopy(model)
        fused_model = fuse_model(fused_model)

        # Print model architecture
        print(f"\nFused Model Architecture:")
        print(fused_model)

        # Load best fused model
        checkpoint = torch.load(args.best_fused_model_path)
        fused_model.load_state_dict(checkpoint['model_state_dict'])

        # Run test on best fused model
        fused_test_loss, fused_test_acc = validate(fused_model, test_loader, criterion, device)
        print(f"Fused Test Loss: {fused_test_loss:.4f}, Fused Test Acc: {fused_test_acc:.4f}")
        
        # Count parameters in fused model
        fused_total_params = sum(p.numel() for p in fused_model.parameters())
        fused_trainable_params = sum(p.numel() for p in fused_model.parameters() if p.requires_grad)
        print(f"\nFused Model Parameters:")
        print(f"  Total parameters: {fused_total_params:,}")
        print(f"  Trainable parameters: {fused_trainable_params:,}")
        
        # Log fused model info to wandb
        wandb.log({
            'fused_total_params': fused_total_params,
            'fused_trainable_params': fused_trainable_params,
            'fused_test_loss': fused_test_loss,
            'fused_test_acc': fused_test_acc
        })
        wandb.run.summary['fused_total_params'] = fused_total_params
        wandb.run.summary['fused_trainable_params'] = fused_trainable_params
        wandb.run.summary['fused_test_loss'] = fused_test_loss
        wandb.run.summary['fused_test_acc'] = fused_test_acc
        
    wandb.finish()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
