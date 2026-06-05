import argparse
import sys
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from dataset import MultiConADDataset, collate_fn
from model import ConGrAD
from modules import graphify, sinusoidal_encoding
from utils import Tee


class PretrainConGrAD(ConGrAD):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reconstruct_head = nn.Linear(cfg.d_model, cfg.d_model)
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model))

    def forward(self, audio, text, mask, speakers):
        B, L = audio.shape[:2]
        lengths = mask.sum(dim=1).long()

        audio = self.audio_proj(audio)
        text  = self.text_proj(text)
        
        for layer in self.co_attn:
            audio, text = layer(audio, text, ~mask)

        h = self.fusion(torch.cat([audio, text], dim=-1))
        h = h + sinusoidal_encoding(L, h.size(-1), h.device)

        # Detach targets to prevent gradient leakage
        h_target = h.clone().detach()
        node_mask = (torch.rand(B, L, device=h.device) < 0.15) & mask
        h_masked = h.clone()
        h_masked[node_mask] = self.mask_token

        x, edge_index, edge_type, batch, _ = graphify(
            h_masked, lengths, speakers, self.window, h.device, drop_edge=self.drop_edge
        )
        x = self.rgat(x, edge_index, edge_type)

        x_reshaped = torch.zeros(B, L, x.size(-1), device=x.device)
        for b in range(B):
            x_reshaped[b, :lengths[b]] = x[batch == b]

        return self.reconstruct_head(x_reshaped), h_target, node_mask


def plot_pretrain_training(history: dict, path: Path, experiment_name: str):
    epochs = range(1, len(history['train_loss']) + 1)
    checkpoints = history['checkpoints']

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    
    for ep in checkpoints:
        plt.scatter(ep, history['train_loss'][ep - 1],
                    color='green', marker='*', s=180, zorder=5,
                    label='Checkpoint' if ep == checkpoints[0] else '')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Pretraining Loss: {experiment_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pretraining training plot saved to {path}")


def pretrain(cfg, device, run_dir: Path):
    ds = MultiConADDataset(cfg.train_jsonl, cfg.gemma_dir, cfg.trill_dir, cfg.speakers_json)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = PretrainConGrAD(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {'train_loss': [], 'checkpoints': []}

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        
        for trill, gemma, speakers, mask, _, _ in pbar:
            trill = trill.to(device)
            gemma = gemma.to(device)
            speakers = speakers.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            pred, target, node_mask = model(trill, gemma, mask, speakers)
            
            loss = F.mse_loss(pred[node_mask], target[node_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(loader)
        history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), run_dir / f'epoch_{epoch}.pt')
            history['checkpoints'].append(epoch)
            print(f"  ↳ checkpoint saved (epoch {epoch})")
            
    torch.save(model.state_dict(), run_dir / 'last_checkpoint.pt')
    plot_pretrain_training(history, run_dir / 'training.png', cfg.experiment_name)
    print(f"\nPretraining finished. Checkpoints saved to {run_dir}")


def main(config_path: str):
    cfg     = Config.from_yaml(config_path)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(cfg.results_dir) / f"{ts}_{cfg.experiment_name}_pretrain"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / 'cfg_pretrain.yaml')

    log_file   = open(run_dir / 'output.log', 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Run dir: {run_dir}")
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    pretrain(cfg, device, run_dir)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg_pretrain.yaml')
    args = parser.parse_args()
    main(args.config)


