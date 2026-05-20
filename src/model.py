import torch
import torch.nn as nn
from torch_geometric.nn import RGATConv
import torch.nn.functional as F


def graphify(h, lengths, speakers, window, device, log_base=2):
    """
    Transform a batch of utterance sequences into PyG graphs.

    PyG MessagePassing modules (e.g. RGATConv) do not support batching,
    so we output a single graph derived of disconnected sub-graphs.
 
    Args:
        h              [B, L_max, d_h]  utterance embeddings (post positional encoding)
        lengths        [B]              true sequence length per sample
        speakers [B, L_max]             speaker ID per utterance (0=interviewer, 1=participant)
        window         int              symmetric local context window (past and future)
        device                          target device
        log_base       int              base for logarithmic skip edges (default: 2)
 
    Returns:
        x            [N, d_h]  flat node features
        edge_index   [2, E]    global edge array
        edge_type    [E]       relation index in {0,1,2,3}
        batch        [N]       sample index per node
        spk_mask [N]           True for participant utterances
    """
    srcs, dsts, rels = [], [], []
    node_feats, batches, masks = [], [], []
    offset = 0      # for global indexing
 
    for b, L in enumerate(lengths.tolist()):
        spk = speakers[b, :L]
        node_feats.append(h[b, :L])
        batches.append(torch.full((L,), b, dtype=torch.long, device=device))
        masks.append(spk == 1) # convert to bool
 
        # Collect undirected pairs via a set
        pairs = set()
        for i in range(L):
            # Add neighbour window edges
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
            # Add logarithmic jump edges
            step = log_base
            while step < L:
                for j in (i - step, i + step):
                    if 0 <= j < L:
                        pairs.add((min(i, j), max(i, j)))
                step *= log_base
 
        # Convert undirected pairs to directed edges
        for u, v in pairs:
            su, sv = spk[u].item(), spk[v].item()
            srcs += [u + offset, v + offset]
            dsts += [v + offset, u + offset]
            rels += [su * 2 + sv, sv * 2 + su]      # 0: int->int    1: int->par
                                                    # 2: par->int    3: par->par
 
        offset += L

    x = torch.cat(node_feats).to(device)
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    edge_type = torch.tensor(rels, dtype=torch.long, device=device)
    batch = torch.cat(batches)
    spk_mask = torch.cat(masks)
 
    return x, edge_index, edge_type, batch, spk_mask
 

class CoAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        kw = dict(dropout=dropout, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(d_model, n_heads, **kw)
        self.text_attn = nn.MultiheadAttention(d_model, n_heads, **kw)
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.ffn_t = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_t2 = nn.LayerNorm(d_model)

    def forward(self, audio, text, kpm):
        # kpm: True = IGNORE (PyTorch convention), so we pass ~mask
        a, _ = self.audio_attn(audio, text, text, key_padding_mask=kpm)
        t, _ = self.text_attn (text, audio, audio, key_padding_mask=kpm)
        audio = self.norm_a1(audio + a)
        text = self.norm_t1(text + t)
        audio = self.norm_a2(audio + self.ffn_a(audio))
        text = self.norm_t2(text + self.ffn_t(text))
        return audio, text


class RGAT(nn.Module):
    def __init__(self, d_h, n_layers, n_heads, dropout, n_relations=4):
        super().__init__()
        assert d_h % n_heads == 0
        self.layers = nn.ModuleList([
            RGATConv(
                in_channels=d_h,
                out_channels=d_h // n_heads,
                num_relations=n_relations,
                heads=n_heads,
                attention_mechanism="across-relation",
                attention_mode="multiplicative-self-attention",
                dropout=dropout,
                concat=True,
            )
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        for conv in self.layers:
            x = self.drop(F.elu(conv(x, edge_index, edge_type))) + x
        return x


class CoAttentionClassifier(nn.Module):
    """
    A simple baseline classifier consisting of the initial 
    co-attention fusion module and a classification head.
    """
    def __init__(self, cfg):
        super().__init__()
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.text_proj  = nn.Linear(cfg.text_dim,  cfg.d_model)
        self.layers = nn.ModuleList([
            CoAttentionBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )

    def forward(self, audio, text, mask):
        # mask: [B, N] True = valid token
        kpm = ~mask                                # invert for PyTorch MHA
        audio = self.audio_proj(audio)             # [B, N, d]
        text = self.text_proj(text)                # [B, N, d]
        for layer in self.layers:
            audio, text = layer(audio, text, kpm)
        m = mask.unsqueeze(-1).float()             # [B, N, 1]
        audio_pool = (audio * m).sum(1) / m.sum(1) # [B, d]  masked mean
        text_pool = (text  * m).sum(1) / m.sum(1)  # [B, d]
        return self.classifier(torch.cat([audio_pool, text_pool], dim=-1))


