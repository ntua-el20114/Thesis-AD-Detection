import sys, shutil, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from config import Config
from dataset import MultiConADDataset, collate_fn, make_balanced_sampler
from model import CoAttentionClassifier, ConGrAD
from utils import Tee, set_seed, extract_metrics, save_results_csv, \
                  compute_uar, plot_training, TARGET_NAMES, print_model_summary, \
                  Visualizer


MODEL = ConGrAD


def mixup_batch(batch_reg, batch_bal, alpha, n_classes, device):
    """
    Balanced-MixUp (Galdran et al., MICCAI 2021).
      - lam ~ Beta(alpha, 1): skews toward 0, regular batch dominates via (1-lam)
      - Features:  x_mix = (1-lam)*x_reg + lam*x_bal
      - Labels:    y_mix = (1-lam)*y_reg + lam*y_bal  (soft one-hot)
    Loss is computed as soft cross-entropy (see run_epoch).
    """
    trill_r, gemma_r, _,        mask_r, y_r = batch_reg
    trill_b, gemma_b, speakers, mask_b, y_b = batch_bal
    trill_r, gemma_r, mask_r = trill_r.to(device), gemma_r.to(device), mask_r.to(device)
    trill_b, gemma_b, mask_b = trill_b.to(device), gemma_b.to(device), mask_b.to(device)
    speakers, y_r, y_b = speakers.to(device), y_r.to(device), y_b.to(device)

    # Pad both batches to the same sequence length
    T = max(trill_r.size(1), trill_b.size(1))
    def _p3(x): return F.pad(x, (0, 0, 0, T - x.size(1)))
    def _p2(x, v=0): return F.pad(x, (0, T - x.size(1)), value=v)
    trill_r, trill_b = _p3(trill_r), _p3(trill_b)
    gemma_r, gemma_b = _p3(gemma_r), _p3(gemma_b)
    mask_r,  mask_b  = _p2(mask_r),  _p2(mask_b)
    speakers         = _p2(speakers)

    lam   = np.random.beta(alpha, 1)
    y_mix = ((1 - lam) * F.one_hot(y_r, n_classes).float()
             +      lam * F.one_hot(y_b, n_classes).float())
    return ((1-lam)*trill_r + lam*trill_b,
            (1-lam)*gemma_r + lam*gemma_b,
            speakers, mask_b, y_mix)


def run_epoch(model, loader, device, optimizer=None, criterion=None,
              balanced_loader=None, cfg=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, labels = 0, [], []

    loader_iter = zip(loader, balanced_loader) if (training and balanced_loader) \
                  else ((b, None) for b in loader)

    with torch.set_grad_enabled(training):
        for batch_reg, batch_bal in loader_iter:
            if batch_bal is not None:
                trill, gemma, speakers, mask, y = mixup_batch(
                    batch_reg, batch_bal, cfg.mixup_alpha, cfg.n_classes, device
                )
                logits = model(trill, gemma, mask, speakers)
                loss   = -(F.log_softmax(logits, dim=-1) * y).sum(dim=-1).mean()
            else:
                trill, gemma, speakers, mask, y = batch_reg
                trill, gemma, speakers, mask, y = (
                    trill.to(device), gemma.to(device),
                    speakers.to(device), mask.to(device), y.to(device)
                )
                logits = model(trill, gemma, mask, speakers)
                if training:
                    loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend((y.argmax(-1) if y.dim() == 2 else y).cpu().tolist())
    return total_loss / len(loader), preds, labels


def _collect_vis(model, loader, device, visualizer):
    """
    Single eval pass that extracts pooled vectors for t-SNE.
    Uses return_pooled=True; corpus names are read from loader.dataset.corpus_names.
    Called after the best model has been loaded, so loader must use shuffle=False.
    """
    model.eval()
    bs = loader.batch_size
    ds = loader.dataset
    with torch.no_grad():
        for i, (trill, gemma, speakers, mask, y) in enumerate(loader):
            trill, gemma, speakers, mask = (
                trill.to(device), gemma.to(device),
                speakers.to(device), mask.to(device)
            )
            logits, pooled = model(trill, gemma, mask, speakers, return_pooled=True)
            start  = i * bs
            end    = start + y.size(0)
            visualizer.collect(
                pooled,
                y.tolist(),
                logits.argmax(-1).cpu().tolist(),
                ds.corpus_names[start:end],
            )


def train_one_run(cfg, run_dir: Path, device,
                  train_loader, test_loader,
                  balanced_loader, visualizer) -> dict:
    model     = MODEL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_uar': [], 'val_uar': [],
               'checkpoints': [], 'stopped_at': None}
    best_uar, patience_counter = 0.0, 0

    for epoch in range(cfg.epochs):
        loss, train_preds, train_labels = run_epoch(
            model, train_loader, device, optimizer, criterion, balanced_loader, cfg
        )
        _, val_preds, val_labels = run_epoch(model, test_loader, device)

        train_uar = compute_uar(train_labels, train_preds)
        val_uar   = compute_uar(val_labels,   val_preds)
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

    # Visualization
    if visualizer.enabled:
        _collect_vis(model, test_loader, device, visualizer)
        visualizer.plot_tsne(f"{cfg.experiment_name} / {run_dir.name}")
        # Graph: collect up to 4 test samples of reasonable length for 2x2 grid
        ds = test_loader.dataset
        graph_samples = []
        for s in ds.samples:
            spk = torch.tensor(ds.speakers[s['File_Name']])
            if 10 <= len(spk) <= 40:
                graph_samples.append((spk, s['File_Name']))
            if len(graph_samples) == 4:
                break
        if graph_samples:
            visualizer.plot_graph(graph_samples, cfg.graph_window)

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

    balanced_loader = None
    if cfg.balanced_mixup:
        balanced_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                     sampler=make_balanced_sampler(train_ds), **loader_kw)

    # Guard against silent misconfiguration
    if not cfg.balanced_mixup and cfg.mixup_alpha > 0:
        print('WARNING: mixup_alpha is set but balanced_mixup is False — augmentation disabled.')

    print("\n" + "="*50)
    print("EXPERIMENT: ", cfg.experiment_name)
    print("="*50)
    print(cfg.experiment_description, "\n")
    print(f"Balanced-MixUp: {'ON' if cfg.balanced_mixup else 'OFF'}")
    print(f"Visualizations: {'ON' if cfg.visualizations else 'OFF'}\n")
    print_model_summary(MODEL(cfg))

    all_metrics = []
    for i in range(cfg.n_runs):
        set_seed(cfg.base_seed + i)
        print(f"\n{'='*40}\nRun {i+1}/{cfg.n_runs} (seed={cfg.base_seed + i})\n{'='*40}")
        rep_dir = run_dir / f"run_{i}"
        rep_dir.mkdir()
        visualizer = Visualizer(cfg.visualizations, rep_dir)
        all_metrics.append(
            train_one_run(cfg, rep_dir, device,
                          train_loader, test_loader, balanced_loader, visualizer)
        )

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
