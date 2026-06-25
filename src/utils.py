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
    """Sets the base_seed as the global seed in all operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model_summary(model: torch.nn.Module):
    """Prints the model architecture and parameter count."""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE & PARAMETERS")
    print("="*50)
    print(model)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 50)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("="*50 + "\n")


def report_vram_projection(cfg, model_class, train_ds, device):
    """Dynamically projects worst-case VRAM using torchinfo and the dataset's max sequence length."""
    try:
        from torchinfo import summary
        dummy_model = model_class(cfg).to(device)
        
        # Find the absolute longest sequence in the dataset
        max_len = max(len(train_ds.speakers[s['File_Name']]) for s in train_ds.samples)
        
        # Create a worst-case dummy batch
        audio = torch.randn(cfg.batch_size, max_len, cfg.audio_dim, device=device)
        text = torch.randn(cfg.batch_size, max_len, cfg.text_dim, device=device)
        speakers = torch.zeros(cfg.batch_size, max_len, dtype=torch.long, device=device)
        mask = torch.ones(cfg.batch_size, max_len, dtype=torch.bool, device=device)
        
        print("\n" + "="*50)
        print(f"WORST-CASE VRAM PROJECTION (Batch Size: {cfg.batch_size}, Max Seq Len: {max_len})")
        print("="*50)
        summary(dummy_model, input_data=(audio, text, mask, speakers), verbose=1)
        
        # Clean up memory
        del dummy_model, audio, text, speakers, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        print("\n[INFO] `torchinfo` is not installed. Run `pip install torchinfo` to see VRAM projections.\n")


def compute_uar(labels, preds) -> float:
    r: dict[str, Any] = classification_report(labels, preds, target_names=TARGET_NAMES,
                                               output_dict=True, zero_division=0)
    return float(r['macro avg']['recall'])


def extract_metrics(labels, preds, test_domains=None) -> dict:
    r: dict[str, Any] = classification_report(labels, preds, target_names=TARGET_NAMES,
                                               output_dict=True, zero_division=0)
    def f1(key): return round(float(r[key]['f1-score']), 4)
    return {
        'UAR':         round(float(r['macro avg']['recall']),      4),
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
    checkpoints = history['checkpoints']
    stopped_at  = history['stopped_at']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, history['train_loss'], label='Train loss', color='steelblue')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_uar'], label='Train UAR', color='steelblue', linestyle='--', alpha=0.6)
    ax2.plot(epochs, history['val_uar'],   label='Val UAR',   color='darkorange')

    for ep in checkpoints:
        ax2.scatter(ep, history['val_uar'][ep - 1],
                    color='green', marker='*', s=180, zorder=5,
                    label='Checkpoint' if ep == checkpoints[0] else '')
    if stopped_at:
        ax2.axvline(x=stopped_at, color='red', linestyle=':', linewidth=1.5,
                    label=f'Early stop (epoch {stopped_at})')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('UAR')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"{experiment_name} - {path.parent.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training plot saved to {path}")


class Visualizer:
    """
    Produces two diagnostic plots after each run:

      plot_tsne()  — 3-panel t-SNE of the pooled representation (the d_model
                     vector fed directly into the classifier), coloured by:
                       1. true class
                       2. predicted class
                       3. corpus of origin
                     Mismatches between panels 1 and 2 show where the model
                     confuses classes; panel 3 reveals dataset-level clustering,
                     which would indicate the model is learning corpus artefacts
                     rather than pathological speech characteristics.

      plot_graph() — conversation graph topology for one representative sample,
                     showing local-window edges and log-step skip edges,
                     coloured by the four speaker-typed relations.
    """

    # Edge relation colours: int→int, int→par, par→int, par→par
    _EDGE_COLORS = ['#999999', '#3498db', '#e74c3c', '#2ecc71']
    _EDGE_LABELS = ['int→int', 'int→par', 'par→int', 'par→par']

    def __init__(self, enabled: bool, out_dir: Path):
        self.enabled  = enabled
        self.out_dir  = out_dir
        self._pooled  = []
        self._labels  = []
        self._preds   = []
        self._corpus  = []

    def collect(self, pooled: torch.Tensor, labels: list, preds: list, corpus: list):
        """Accumulate one batch of pooled vectors and metadata."""
        if not self.enabled: return
        self._pooled.append(pooled.cpu().numpy())
        self._labels.extend(labels)
        self._preds.extend(preds)
        self._corpus.extend(corpus)

    def plot_tsne(self, title: str = ''):
        """Run t-SNE on accumulated pooled vectors and save a 3-panel figure."""
        if not self.enabled or not self._pooled: return
        from sklearn.manifold import TSNE

        X      = np.concatenate(self._pooled, axis=0)
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=min(30, len(X) - 1)).fit_transform(X)

        labels = np.array(self._labels)
        preds  = np.array(self._preds)
        corpus = np.array(self._corpus)

        cls_cmap  = plt.cm.get_cmap('tab10', len(TARGET_NAMES))
        uniq_corp = sorted(np.unique(corpus))
        corp_cmap = plt.cm.get_cmap('Set2', len(uniq_corp))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panels 1 & 2: true / predicted class
        for ax, values, t in [(axes[0], labels, 'True Labels'),
                               (axes[1], preds,  'Predicted Labels')]:
            for i, name in enumerate(TARGET_NAMES):
                m = values == i
                if m.any():
                    ax.scatter(coords[m, 0], coords[m, 1], c=[cls_cmap(i)],
                               label=name, alpha=0.7, s=18)
            ax.set_title(t)
            ax.legend(fontsize=8)

        # Panel 3: corpus origin
        for i, corp in enumerate(uniq_corp):
            m = corpus == corp
            axes[2].scatter(coords[m, 0], coords[m, 1], c=[corp_cmap(i)],
                            label=corp, alpha=0.7, s=18)
        axes[2].set_title('Source Corpus')
        axes[2].legend(fontsize=7)

        plt.suptitle(f't-SNE of Pooled Representations — {title}')
        plt.tight_layout()
        path = self.out_dir / 'tsne.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  t-SNE saved to {path}')

        # Reset for next run
        self._pooled, self._labels, self._preds, self._corpus = [], [], [], []

    def plot_graph(self, samples: list, window: int):
        """
        2x2 grid of conversation graph topology plots.
        samples: list of (speakers_tensor, file_name) — up to 4 entries.
        Nodes arranged in a circle; edges coloured by speaker-typed relation.
        """
        if not self.enabled: return
        try:
            import networkx as nx
        except ImportError:
            print('  networkx not found — skipping graph plot (pip install networkx)')
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for ax_idx, (speakers, name) in enumerate(samples[:4]):
            ax  = axes[ax_idx]
            L   = len(speakers)
            spk = speakers.tolist()

            # Reconstruct edges (mirrors graphify logic, topology only)
            pairs = set()
            for i in range(L):
                for j in range(max(0, i - window), min(L, i + window + 1)):
                    if j != i:
                        pairs.add((min(i, j), max(i, j)))
                step = 2
                while step < L:
                    for j in (i - step, i + step):
                        if 0 <= j < L:
                            pairs.add((min(i, j), max(i, j)))
                    step *= 2

            G = nx.DiGraph()
            G.add_nodes_from(range(L))
            edge_rel = {}
            for u, v in pairs:
                su, sv = spk[u], spk[v]
                G.add_edge(u, v); G.add_edge(v, u)
                edge_rel[(u, v)] = su * 2 + sv
                edge_rel[(v, u)] = sv * 2 + su

            pos         = nx.circular_layout(G)
            node_colors = ['#3498db' if spk[i] == 0 else '#e74c3c' for i in range(L)]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
            nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(L)},
                                    font_size=7, font_color='white', ax=ax)
            for rel, color in enumerate(self._EDGE_COLORS):
                edges = [e for e, r in edge_rel.items() if r == rel]
                if edges:
                    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color,
                                           alpha=0.35, arrows=True, ax=ax,
                                           connectionstyle='arc3,rad=0.15', arrowsize=8)
            ax.set_title(name, fontsize=9)
            ax.axis('off')

        from matplotlib.patches import Patch
        legend = (
            [Patch(fc='#3498db', label='Interviewer'),
             Patch(fc='#e74c3c', label='Participant')] +
            [Patch(fc=c, label=l) for c, l in zip(self._EDGE_COLORS, self._EDGE_LABELS)]
        )
        fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=8,
                   bbox_to_anchor=(0.5, 0.01))
        plt.suptitle(f'Conversation Graphs  (window={window}, log-step skip edges)', fontsize=11)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        path = self.out_dir / 'graph.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Graph saved to {path}')


def get_alpha(epoch: int, total_epochs: int) -> float:
    """
    DANN annealing schedule (Ganin et al., JMLR 2016).
    Ramps the GRL coefficient from 0 → ~1 using a sigmoid curve,
    letting the diagnosis classifier stabilise before the adversary activates.
    """
    import math
    p = epoch / max(total_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
