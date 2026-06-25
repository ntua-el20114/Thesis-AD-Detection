import sys, argparse, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, f1_score

from config import Config
from modules import CoAttentionBlock, RGAT, sinusoidal_encoding
from utils import Tee, set_seed

MELD_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
EMOTION_MAP = {e: i for i, e in enumerate(MELD_EMOTIONS)}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MELDDataset(Dataset):
    def __init__(self, jsonl_path, gemma_dir, trill_dir):
        # jsonl where each line is a dialogue: 
        # {"Dialogue_ID": "dia0", "Speakers": [0, 1, 0, 2], "Emotions": ["anger", "joy", "neutral", "sadness"]}
        self.gemma_dir = Path(gemma_dir)
        self.trill_dir = Path(trill_dir)
        
        jsonl_path = Path(jsonl_path).expanduser()
        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found.")
            self.dialogues = []
        else:
            with open(jsonl_path) as f:
                self.dialogues = [json.loads(l) for l in f]
                
        valid_dialogues = []
        for d in self.dialogues:
            name = d["Dialogue_ID"]
            if (self.trill_dir / f"{name}.pt").exists() and (self.gemma_dir / f"{name}.pt").exists():
                valid_dialogues.append(d)
        self.dialogues = valid_dialogues

    def __getitem__(self, idx):
        d = self.dialogues[idx]
        name = d["Dialogue_ID"]
        trill = torch.load(self.trill_dir / f"{name}.pt")
        gemma = torch.load(self.gemma_dir / f"{name}.pt")
        speakers = torch.tensor(d["Speakers"], dtype=torch.long)
        labels = torch.tensor([EMOTION_MAP[e] for e in d["Emotions"]], dtype=torch.long)
        return trill, gemma, speakers, labels

    def __len__(self):
        return len(self.dialogues)


def meld_collate_fn(batch):
    trills, gemmas, speakers, labels = zip(*batch)
    lengths      = torch.tensor([t.size(0) for t in trills])
    trill_pad    = pad_sequence(trills,   batch_first=True)
    gemma_pad    = pad_sequence(gemmas,   batch_first=True)
    speakers_pad = pad_sequence(speakers, batch_first=True, padding_value=0)
    labels_pad   = pad_sequence(labels,   batch_first=True, padding_value=-100)
    mask         = torch.arange(trill_pad.size(1)) < lengths.unsqueeze(1)
    return trill_pad, gemma_pad, speakers_pad, mask, labels_pad


class DummyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
    def __len__(self): 
        return 16
    def __getitem__(self, idx):
        L = torch.randint(3, 10, (1,)).item()
        trill = torch.randn(L, self.cfg.audio_dim)
        gemma = torch.randn(L, self.cfg.text_dim)
        speakers = torch.randint(0, 3, (L,))
        labels = torch.randint(0, len(MELD_EMOTIONS), (L,))
        return trill, gemma, speakers, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def meld_graphify(h, lengths, speakers, window, device, log_base=2, drop_edge=0.0):
    srcs, dsts, rels = [], [], []
    node_feats, batches = [], []
    offset = 0

    for b, L in enumerate(lengths.tolist()):
        spk = speakers[b, :L]
        node_feats.append(h[b, :L])
        batches.append(torch.full((L,), b, dtype=torch.long, device=device))

        pairs = set()
        for i in range(L):
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
            step = log_base
            while step < L:
                for j in (i - step, i + step):
                    if 0 <= j < L:
                        pairs.add((min(i, j), max(i, j)))
                step *= log_base

        for u, v in pairs:
            su, sv = spk[u].item(), spk[v].item()
            srcs += [u + offset, v + offset]
            dsts += [v + offset, u + offset]
            
            # Directional relations:
            # 0: same speaker, past   (u < v, so u is in the past of v)
            # 1: same speaker, future (v > u, so v is in the future of u)
            # 2: diff speaker, past   
            # 3: diff speaker, future 
            
            if su == sv:
                rels += [0 if u < v else 1, 1 if u < v else 0]
            else:
                rels += [2 if u < v else 3, 3 if u < v else 2]

        offset += L

    x          = torch.cat(node_feats).to(device)
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    edge_type  = torch.tensor(rels,         dtype=torch.long, device=device)
    batch      = torch.cat(batches)

    if drop_edge > 0.0:
        keep       = torch.rand(edge_index.size(1), device=device) > drop_edge
        edge_index = edge_index[:, keep]
        edge_type  = edge_type[keep]

    return x, edge_index, edge_type, batch



class MELDConGrAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.text_proj  = nn.Linear(cfg.text_dim,  cfg.d_model)
        self.co_attn = nn.ModuleList([
            CoAttentionBlock(cfg.d_model, cfg.ca_heads, cfg.dropout)
            for _ in range(cfg.rgat_layers)
        ])
        self.fusion    = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.rgat      = RGAT(cfg.d_model, cfg.rgat_layers, cfg.rgat_heads, cfg.dropout, n_relations=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        self.window    = cfg.graph_window
        self.drop_edge = cfg.drop_edge

    def forward(self, audio, text, mask, speakers):
        B, L    = audio.shape[:2]
        lengths = mask.sum(dim=1).long()

        audio = self.audio_proj(audio)
        text  = self.text_proj(text)
        for layer in self.co_attn:
            audio, text = layer(audio, text, ~mask)

        # 1. Fuse features (MERC-GCN BiGRU equivalent)
        h = self.fusion(torch.cat([audio, text], dim=-1))
        h = h + sinusoidal_encoding(L, h.size(-1), h.device)

        # 2. Extract 2D node features
        h_flat, edge_index, edge_type, batch = meld_graphify(
            h, lengths, speakers, self.window, h.device, drop_edge=self.drop_edge
        )
        
        # 3. Pass through RGAT
        x = self.rgat(h_flat, edge_index, edge_type)

        # 4. Concatenate pre-graph and post-graph features
        x_combined = torch.cat([h_flat, x], dim=-1)

        # 5. Classify
        logits_flat = self.classifier(x_combined)
        logits = torch.zeros(B, L, logits_flat.size(-1), device=x.device)
        offset = 0
        for b, l in enumerate(lengths.tolist()):
            logits[b, :l] = logits_flat[offset:offset+l]
            offset += l
            
        return logits


class MELDCoAttentionClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.text_proj  = nn.Linear(cfg.text_dim,  cfg.d_model)
        self.layers = nn.ModuleList([
            CoAttentionBlock(cfg.d_model, cfg.ca_heads, cfg.dropout)
            for _ in range(cfg.ca_layers)
        ])
        self.fusion = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )

    def forward(self, audio, text, mask, speakers=None):
        kpm = ~mask
        audio = self.audio_proj(audio)
        text = self.text_proj(text)
        for layer in self.layers:
            audio, text = layer(audio, text, kpm)
        
        pooled = self.fusion(torch.cat([audio, text], dim=-1))
        logits = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, cfg, optimizer=None, criterion=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, labels_list = 0, [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            audio, text, speakers, mask, y = batch
            audio, text, speakers, mask, y = (
                audio.to(device), text.to(device), speakers.to(device),
                mask.to(device), y.to(device)
            )
            
            logits = model(audio, text, mask, speakers) # [B, L, n_classes]
            
            logits_flat = logits.view(-1, cfg.n_classes)
            y_flat = y.view(-1)
            
            valid_mask = y_flat != -100
            
            if valid_mask.sum() > 0:
                loss = criterion(logits_flat[valid_mask], y_flat[valid_mask])
                
                if training:
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    total_loss += loss.item()
                    
                preds.extend(logits_flat[valid_mask].argmax(-1).cpu().tolist())
                labels_list.extend(y_flat[valid_mask].cpu().tolist())

    return total_loss / max(len(loader), 1), preds, labels_list


def train_meld(cfg, run_dir, device, train_loader, test_loader, model_class):
    model = model_class(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best_f1, patience_counter = 0.0, 0

    for epoch in range(cfg.epochs):
        loss, train_preds, train_labels = run_epoch(
            model, train_loader, device, cfg, optimizer, criterion
        )
        _, val_preds, val_labels = run_epoch(
            model, test_loader, device, cfg, criterion=criterion
        )

        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | loss {loss:.4f} | "
              f"train F1 {train_f1 * 100:.2f} % | val F1 {val_f1 * 100:.2f} %")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pt')
            print(f"    ↳ best model saved (F1 {val_f1 * 100:.2f} %)")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(run_dir / 'best_model.pt', weights_only=True))
    _, test_preds, test_labels = run_epoch(model, test_loader, device, cfg, criterion=criterion)
    print("\nTest Results:")
    print(classification_report(test_labels, test_preds, target_names=MELD_EMOTIONS, zero_division=0))


def main(config_path: str, model_type: str):
    cfg = Config.from_yaml(config_path)
    cfg.n_classes = len(MELD_EMOTIONS) 
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(cfg.results_dir).expanduser() / f"{ts}_MELD_{model_type}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file   = open(run_dir / 'output.log', 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = MELDDataset(cfg.train_jsonl, cfg.gemma_dir, cfg.trill_dir)
    test_ds  = MELDDataset(cfg.test_jsonl,  cfg.gemma_dir, cfg.trill_dir)
    num_workers = 4
    
    if len(train_ds) == 0:
        print("Using Dummy Dataset for testing since actual dataset was not found.")
        train_ds = DummyDataset(cfg)
        test_ds = DummyDataset(cfg)
        num_workers = 0

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=meld_collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, 
                              collate_fn=meld_collate_fn, num_workers=num_workers)

    model_class = MELDConGrAD if model_type == 'ConGrAD' else MELDCoAttentionClassifier

    print(f"Starting training on MELD with {model_type}...")
    n_runs = getattr(cfg, 'n_runs', 1)
    for i in range(n_runs):
        set_seed(cfg.base_seed + i)
        print(f"\n{'='*40}\nRun {i+1}/{n_runs} (seed={cfg.base_seed + i})\n{'='*40}")
        rep_dir = run_dir / f"run_{i}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        train_meld(cfg, rep_dir, device, train_loader, test_loader, model_class)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, choices=['ConGrAD', 'CoAttention'], default='ConGrAD')
    args = parser.parse_args()
    main(args.config, args.model)
