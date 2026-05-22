import sys, shutil, argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from config import Config
from dataset import MultiConADDataset, collate_fn
from model import CoAttentionClassifier, ConGrAD
from utils import Tee, set_seed, extract_metrics, save_results_csv, \
                  compute_uar, plot_training, TARGET_NAMES, print_model_summary


MODEL = ConGrAD


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.set_grad_enabled(training):
        for trill, gemma, speakers, mask, y in loader:
            trill, gemma, speakers, mask, y = (
                trill.to(device), gemma.to(device),
                speakers.to(device), mask.to(device), y.to(device)
            )
            logits = model(trill, gemma, mask, speakers)
            if training:
                loss = criterion(logits, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    return total_loss / len(loader), preds, labels


def train_one_run(cfg, run_dir: Path, device, train_loader, test_loader) -> dict:
    model = MODEL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_uar': [], 'val_uar': [],
               'checkpoints': [], 'stopped_at': None}
    best_uar, patience_counter = 0.0, 0

    for epoch in range(cfg.epochs):
        loss, train_preds, train_labels = run_epoch(model, train_loader, device, optimizer, criterion)
        _, val_preds, val_labels = run_epoch(model, test_loader, device)

        train_uar = compute_uar(train_labels, train_preds)
        val_uar = compute_uar(val_labels, val_preds)

        history['train_loss'].append(loss)
        history['train_uar'].append(train_uar)
        history['val_uar'].append(val_uar)

        print(f"  Epoch {epoch+1:02d}/{cfg.epochs} | loss {loss:.4f} | "
              f"train UAR {train_uar:.4f} | val UAR {val_uar:.4f}")

        torch.save(model.state_dict(), run_dir / 'last_checkpoint.pt')

        if val_uar > best_uar:
            best_uar = val_uar
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pt')
            history['checkpoints'].append(epoch + 1)
            print(f"    ↳ best model saved (UAR {val_uar:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                history['stopped_at'] = epoch + 1
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    plot_training(history, run_dir / 'training.png', cfg.experiment_name)

    model.load_state_dict(torch.load(run_dir / 'best_model.pt'))
    _, preds, labels = run_epoch(model, test_loader, device)
    print(classification_report(labels, preds, target_names=TARGET_NAMES))
    return extract_metrics(labels, preds)


def main(config_path: str):
    cfg     = Config.from_yaml(config_path)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(cfg.results_dir) / f"{ts}_{cfg.experiment_name}"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / 'config.yaml')

    log_file   = open(run_dir / 'output.log', 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Run dir: {run_dir}")
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    train_ds = MultiConADDataset(cfg.train_jsonl, cfg.gemma_dir, cfg.trill_dir, cfg.speakers_json)
    test_ds  = MultiConADDataset(cfg.test_jsonl,  cfg.gemma_dir, cfg.trill_dir, cfg.speakers_json)
    loader_kw    = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    # print experiment info
    print("\n" + "="*50)
    print("EXPERIMENT: ", cfg.experiment_name)
    print("="*50)
    print(cfg.experiment_description, "\n")

    # print model summary on a temporary sample model
    print_model_summary(MODEL(cfg))

    # Repeat train/eval experiment [n_runs] times
    all_metrics = []
    for i in range(cfg.n_runs):
        set_seed(cfg.base_seed + i)
        print(f"\n{'='*40}\nRun {i+1}/{cfg.n_runs} (seed={cfg.base_seed + i})\n{'='*40}")
        rep_dir = run_dir / f"run_{i}"
        rep_dir.mkdir()
        all_metrics.append(train_one_run(cfg, rep_dir, device, train_loader, test_loader))

    save_results_csv(all_metrics, run_dir / 'results.csv', cfg.base_seed)
    print(f"\nResults saved to {run_dir / 'results.csv'}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)
    print("\n")
