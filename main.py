import json
import torch
import csv
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataloader import create_dataloaders
from src.models.wavlm_model import WavLMClassifier
from src.models.test_models import LinearFusionModel
from src.train import train
import argparse

SAMPLE_RATE = 16000
BASE_SEED = 42

def generate_seeds(num_seeds: int, base_seed: str = BASE_SEED) -> list:
    """
    Generate seeds deterministically based on a base seed string.
    
    Args:
        num_seeds: Number of seeds to generate
        base_seed: Base seed string (uses current datetime if None)
    
    Returns:
        List of integer seeds
    """
    
    seeds = []
    for i in range(num_seeds):
        # Create a deterministic seed by hashing the base seed + index
        hash_input = f"{base_seed}_{i}".encode()
        hash_obj = hashlib.sha256(hash_input)
        seed = int(hash_obj.hexdigest(), 16) % (2**31)  # Keep it within 32-bit range
        seeds.append(seed)
    
    return seeds

def main(
    # model_name: str = 'wavlm',
    # batch_size: int = 4,
    # num_epochs: int = 5,
    # learning_rate: float = 2e-5,
    model: str,
    comment: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    num_repetitions: int = 5,
    val_split: float = 0.1,
    results_dir: str = 'results',
):
    """
    Main training script with multiple experiment repetitions.
    
    Args:
        model_name: Name of model to use
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        num_repetitions: Number of experiment repetitions (default=5)
        val_split: Fraction of training data to use for validation (default=0.1)
        results_dir: Directory to save results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    data_dir = Path("data/MultiConAD")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_jsonl=data_dir / "train_English.jsonl",
        test_jsonl=data_dir / "test_English.jsonl",
        audio_dir=data_dir / "Audio",
        batch_size=batch_size,
        sample_rate=SAMPLE_RATE,
        val_split=val_split,
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{model}_{timestamp}"
    exp_dir = Path(results_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate seeds deterministically
    seeds = generate_seeds(num_repetitions, base_seed=timestamp)
    
    # Save hyperparameters
    config = {
        'model': model,
        'comment': comment,
        'num_repetitions': num_repetitions,
        'val_split': val_split,
        'seeds': seeds,
        'device': device,
        'timestamp': timestamp,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment: {exp_name}")
    
    # Track metrics across repetitions
    all_metrics = []
    
    # Run multiple repetitions
    for rep in range(num_repetitions):
        msg = f"Repetition {rep+1}/{num_repetitions}"
        print(f"\n{'='*len(msg)}")
        print(msg)
        print(f"{'='*len(msg)}")
        
        # Set seed for reproducibility
        seed = seeds[rep]
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Seed: {seed}")

        # Create model for this repetition
        print(f"Creating {model} model...")
        if model == 'wavlm':
            model = WavLMClassifier(num_classes=3)
        elif model == 'test_linear':
            model = LinearFusionModel()
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Train
        print("Starting training...")
        model, history, test_metrics = train(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        
        # Save history for this repetition
        rep_history_file = exp_dir / f'history_rep{rep+1:03d}.json'
        with open(rep_history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Extract final metrics from this repetition
        final_accuracy = test_metrics['test_accuracy']
        
        # For now, we'll use accuracy as primary metric
        # F1 and AUR would need to be computed with actual predictions
        metrics = {
            'repetition': rep + 1,
            'seed': seed,
            'accuracy': final_accuracy,
            'f1': final_accuracy,  
            'aur': final_accuracy,  
        }
        all_metrics.append(metrics)
        
        print(f"Rep {rep+1} - Accuracy: {final_accuracy:.4f}")
        
        # Save model for this repetition
        model_file = exp_dir / f'model_rep{rep+1:03d}.pt'
        torch.save(model.state_dict(), model_file)
    
    # Save metrics CSV with averages and standard deviations
    if all_metrics:
        csv_file = exp_dir / 'metrics.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['repetition', 'seed', 'accuracy', 'f1', 'aur'])
            writer.writeheader()
            writer.writerows(all_metrics)
            
            # Calculate statistics
            accuracies = [m['accuracy'] for m in all_metrics]
            f1_scores = [m['f1'] for m in all_metrics]
            aur_scores = [m['aur'] for m in all_metrics]
            
            # Add averages row
            avg_metrics = {
                'repetition': 'AVERAGE',
                'seed': '-',
                'accuracy': np.mean(accuracies),
                'f1': np.mean(f1_scores),
                'aur': np.mean(aur_scores),
            }
            writer.writerow(avg_metrics)
            
            # Add std deviations row
            std_metrics = {
                'repetition': 'STD_DEV',
                'seed': '-',
                'accuracy': np.std(accuracies),
                'f1': np.std(f1_scores),
                'aur': np.std(aur_scores),
            }
            writer.writerow(std_metrics)
    
    msg = f"Results saved to {exp_dir}"
    print(f"\n{'='*len(msg)}")
    print(msg)
    print(f"{'='*len(msg)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--comment', type=str, default='none', help='Personal comment for the experiment')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--num_repetitions', type=int, default=5, help='Number of experiment repetitions')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    main(
        model=args.model,
        comment=args.comment,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_repetitions=args.num_repetitions,
        val_split=args.val_split,
        results_dir=args.results_dir,
    )
