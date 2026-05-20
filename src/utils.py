import sys, csv, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

TARGET_NAMES = ['HC', 'MCI', 'Dementia']


class Tee:
    """ 
    Copies output from a stream (stdout) to an output file.
    (a bit like GNU Tee)
    This is done to save the console output to a log file.
    """
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


def set_seed(seed: int):
    """
    Sets the base_seed as the global seed in all operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model_summary(model: torch.nn.Module):
    """
    Prints the model architecture and parameter count.
    """
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE & PARAMETERS")
    print("="*50)
    
    # Print architecture
    print(model)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("="*50 + "\n")


def compute_uar(labels, preds) -> float:
    r: dict[str, Any] = classification_report(labels, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    return float(r['macro avg']['recall'])


def extract_metrics(labels, preds) -> dict:
    r: dict[str, Any] = classification_report(labels, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)

    def f1(key: str) -> float:
        return round(float(r[key]['f1-score']), 4)

    return {
        'UAR':         round(float(r['macro avg']['recall']),      4),  # primary metric
        'accuracy':    round(float(r['accuracy']),                 4),
        'HC_f1':       f1('HC'),
        'MCI_f1':      f1('MCI'),
        'Dementia_f1': f1('Dementia'),
        'macro_f1':    round(float(r['macro avg']['f1-score']),    4),
        'weighted_f1': round(float(r['weighted avg']['f1-score']), 4),
    }


def save_results_csv(all_metrics: list, path: Path, base_seed: int):
    keys = list(all_metrics[0].keys())
    rows = {f"run_{i}": {'seed': base_seed + i, **m} for i, m in enumerate(all_metrics)}
    rows['mean'] = {'seed': '-', **{k: round(float(np.mean([m[k] for m in all_metrics])), 4) for k in keys}}
    rows['std']  = {'seed': '-', **{k: round(float(np.std( [m[k] for m in all_metrics])), 4) for k in keys}}
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'seed'] + keys)
        w.writeheader()
        for run_name, metrics in rows.items():
            w.writerow({'run': run_name, **metrics}) 


def plot_training(history: dict, path: Path, experiment_name):
    epochs      = range(1, len(history['train_loss']) + 1)
    checkpoints = history['checkpoints']   # list of epoch numbers (1-indexed)
    stopped_at  = history['stopped_at']    # epoch number or None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Loss ---
    ax1.plot(epochs, history['train_loss'], label='Train loss', color='steelblue')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- UAR ---
    ax2.plot(epochs, history['train_uar'], label='Train UAR', color='steelblue',  linestyle='--', alpha=0.6)
    ax2.plot(epochs, history['val_uar'],   label='Val UAR',   color='darkorange')

    # Mark every checkpoint with a star on the val UAR curve
    for ep in checkpoints:
        ax2.scatter(ep, history['val_uar'][ep - 1],
                    color='green', marker='*', s=180, zorder=5,
                    label='Checkpoint' if ep == checkpoints[0] else '')

    # Mark early stopping with a vertical line
    if stopped_at:
        ax2.axvline(x=stopped_at, color='red', linestyle=':', linewidth=1.5,
                    label=f'Early stop (epoch {stopped_at})')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('UAR')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"{experiment_name} - {path.parent.name}")   # e.g. "run_0"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training plot saved to {path}")
