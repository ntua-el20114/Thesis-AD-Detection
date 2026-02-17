import json
import torch
import csv
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse

from dataloader import create_dataloaders
from models.pretrained import WavLMClassifier
from models.pretrained import TRILLssonClassifier
from models.test_models import LinearFusionModel
from train import train

SAMPLE_RATE = 16000
BASE_SEED = 42

def get_model_info(model):
    """Get model parameter count and architecture string."""
    num_params = sum(p.numel() for p in model.parameters())
    architecture = str(model)
    return num_params, architecture

def generate_seeds(num_seeds: int, base_seed: int = BASE_SEED) -> list:
    """Generate seeds deterministically based on a base seed string."""
    seeds = []
    for i in range(num_seeds):
        hash_input = f"{base_seed}_{i}".encode()
        hash_obj = hashlib.sha256(hash_input)
        seed = int(hash_obj.hexdigest(), 16) % (2**31)
        seeds.append(seed)
    return seeds

def main(
    model_name: str,
    comment: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    num_repetitions: int = 5,
    val_split: float = 0.1,
    results_dir: str = 'results',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    data_dir = Path("data/MultiConAD")
    
    # Pass model_name to create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_jsonl=data_dir / "train_English.jsonl",
        test_jsonl=data_dir / "test_English.jsonl",
        audio_dir=data_dir / "Audio",
        batch_size=batch_size,
        sample_rate=SAMPLE_RATE,
        val_split=val_split,
        model_name=model_name 
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{model_name}_{timestamp}"
    exp_dir = Path(results_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate seeds deterministically
    seeds = generate_seeds(num_repetitions, base_seed=BASE_SEED)
    
    # Create a model instance to get architecture and parameter count
    if model_name == 'test_linear':
        sample_model = LinearFusionModel()
    elif model_name == 'trillsson':
        sample_model = TRILLssonClassifier()
    elif model_name == 'wavlm':
        sample_model = WavLMClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    num_params, architecture = get_model_info(sample_model)
    
    # Save hyperparameters
    config = {
        'model': model_name,
        'comment': comment,
        'num_repetitions': num_repetitions,
        'val_split': val_split,
        'seeds': seeds,
        'device': device,
        'timestamp': timestamp,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_parameters': num_params,
        'model_architecture': architecture,
    }
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model architecture
    with open(exp_dir / 'model_architecture.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Parameters: {num_params:,}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in sample_model.parameters() if p.requires_grad):,}\n\n")
        f.write("Architecture:\n")
        f.write("=" * 80 + "\n")
        f.write(architecture)
    
    # Sample model no longer needed
    del sample_model

    print(f"Experiment: {exp_name}")
    
    # Track metrics across repetitions
    all_metrics = []
    
    # Run multiple repetitions
    for rep in range(num_repetitions):
        msg = f"Repetition {rep+1}/{num_repetitions}"
        print(f"\n{'='*len(msg)}")
        print(msg)
        print(f"{'='*len(msg)}")
        
        seed = seeds[rep]
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Seed: {seed}")

        print(f"Creating {model_name} model...")
        if model_name == 'test_linear':
            current_model = LinearFusionModel()
        elif model_name == 'trillsson':
            current_model = TRILLssonClassifier()
        elif model_name == 'wavlm':
            current_model = WavLMClassifier()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print("Starting training...")
        current_model, history, test_metrics = train(
            current_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        
        rep_history_file = exp_dir / f'history_rep{rep+1:03d}.json'
        with open(rep_history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        metrics = {
            'repetition': rep + 1,
            'seed': seed,
            'accuracy': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'uar': test_metrics['uar'],
        }
        all_metrics.append(metrics)
        
        print(f"Rep {rep+1} - Accuracy: {test_metrics['acc']:.4f}")
        
        model_file = exp_dir / f'model_rep{rep+1:03d}.pt'
        torch.save(current_model.state_dict(), model_file)

        del current_model
    
    # Save metrics CSV
    if all_metrics:
        csv_file = exp_dir / 'metrics.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['repetition', 'seed', 'accuracy', 'f1', 'uar'])
            writer.writeheader()
            writer.writerows(all_metrics)
            
            accuracies = [m['accuracy'] for m in all_metrics]
            f1_scores = [m['f1'] for m in all_metrics]
            uar_scores = [m['uar'] for m in all_metrics]
            
            avg_metrics = {
                'repetition': 'AVERAGE',
                'seed': '-',
                'accuracy': np.mean(accuracies),
                'f1': np.mean(f1_scores),
                'uar': np.mean(uar_scores),
            }
            writer.writerow(avg_metrics)
            
            std_metrics = {
                'repetition': 'STD_DEV',
                'seed': '-',
                'accuracy': np.std(accuracies),
                'f1': np.std(f1_scores),
                'uar': np.std(uar_scores),
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
        model_name=args.model,
        comment=args.comment,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_repetitions=args.num_repetitions,
        val_split=args.val_split,
        results_dir=args.results_dir,
    )
