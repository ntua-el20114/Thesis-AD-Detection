import argparse, shutil, sys
from datetime import datetime
from pathlib import Path

import torch
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from config import Config
from dataset import MultiConADDataset, collate_fn
from model import CoAttentionClassifier
from train import run_epoch
from utils import set_seed, compute_uar, TARGET_NAMES

MODEL = CoAttentionClassifier

SEARCH_SPACE = {
    'lr':       ('float_log', 1e-5, 1e-3),
    'd_model':  ('categorical', [128, 256, 512]),
    'n_heads':  ('categorical', [2, 4, 8]),
    'n_layers': ('categorical', [1, 2, 3]),
    'dropout':  ('float', 0.1, 0.5),
}


def sample_config(trial: optuna.Trial, base_cfg: Config) -> Config:
    overrides = {}
    for name, spec in SEARCH_SPACE.items():
        match spec[0]:
            case 'float_log':
                overrides[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            case 'float':
                overrides[name] = trial.suggest_float(name, spec[1], spec[2])
            case 'categorical':
                overrides[name] = trial.suggest_categorical(name, spec[1])
    # n_heads must divide d_model
    while base_cfg.d_model % overrides.get('n_heads', base_cfg.n_heads) != 0:
        overrides['n_heads'] = trial.suggest_categorical('n_heads', SEARCH_SPACE['n_heads'][1])
    return Config(**{**base_cfg.__dict__, **overrides})


def objective(trial, base_cfg, train_loader, test_loader, device):
    cfg = sample_config(trial, base_cfg)
    set_seed(base_cfg.base_seed)

    model     = MODEL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_uar  = 0.0

    for epoch in range(base_cfg.hpo_epochs):
        run_epoch(model, train_loader, device, optimizer, criterion)
        _, val_preds, val_labels = run_epoch(model, test_loader, device)
        uar = compute_uar(val_labels, val_preds)

        trial.report(uar, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_uar = max(best_uar, uar)

    return best_uar


def main(config_path: str):
    base_cfg = Config.from_yaml(config_path)
    device   = torch.device(base_cfg.device if torch.cuda.is_available() else 'cpu')

    # Output dir
    if args.resume:
        out_dir = Path(args.resume).expanduser().resolve()
    else:
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(base_cfg.results_dir) / f"{ts}_hpo_{base_cfg.experiment_name}"
        out_dir.mkdir(parents=True)
        shutil.copy(config_path, out_dir / 'config.yaml')

    train_ds = MultiConADDataset(base_cfg.train_jsonl, base_cfg.gemma_dir, base_cfg.trill_dir)
    test_ds = MultiConADDataset(base_cfg.test_jsonl,  base_cfg.gemma_dir, base_cfg.trill_dir)
    loader_kw = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=base_cfg.batch_size, shuffle=True,  **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=base_cfg.batch_size, shuffle=False, **loader_kw)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=base_cfg.base_seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name='baseline_hpo',
        storage=f'sqlite:///{out_dir}/hpo.db',   # persists results; resumable
        load_if_exists=True,
    )

    remaining = base_cfg.hpo_trials - len(study.trials)
    if remaining <= 0:
        print(f"Study already has {len(study.trials)} trials, skipping optimization.")
    else:
        study.optimize(
            lambda trial: objective(trial, base_cfg, train_loader, test_loader, device),
            n_trials=remaining,
            show_progress_bar=True,
        )

    # Report
    best = study.best_trial
    print(f"\nBest UAR:    {best.value:.4f}")
    print(f"Best params: {best.params}")

    # Save best params as a ready-to-use config
    best_cfg_data = {**vars(base_cfg), **best.params}
    import yaml
    with open(out_dir / 'best_config.yaml', 'w') as f:
        yaml.dump(best_cfg_data, f, default_flow_style=False)
    print(f"\nBest config saved to {out_dir / 'best_config.yaml'}")

    # Optuna plots
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_contour,
        )
        plot_optimization_history(study).write_html(str(out_dir / 'opt_history.html'))
        plot_param_importances(study).write_html(str(out_dir / 'param_importances.html'))
        plot_contour(study).write_html(str(out_dir / 'contour.html'))
        print(f"Plots saved to {out_dir}")
    except Exception as e:
        print(f"Could not generate Optuna plots: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args.config)
