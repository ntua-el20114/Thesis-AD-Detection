import torch
import torch.nn as nn
from modules import *


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
            CoAttentionBlock(cfg.d_model, cfg.ca_heads, cfg.dropout)
            for _ in range(cfg.ca_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        # self.domain_classifier = DomainClassifier(2 * cfg.d_model, cfg.n_domains, cfg.dropout)

    def forward(self, audio, text, mask, speakers=None, alpha=1.0, return_pooled=False):
        # speakers unused; accepted for interface compatibility with ConGrAD
        kpm        = ~mask
        audio      = self.audio_proj(audio)
        text       = self.text_proj(text)
        for layer in self.layers:
            audio, text = layer(audio, text, kpm)
        m          = mask.unsqueeze(-1).float()
        audio_pool = (audio * m).sum(1) / m.sum(1)
        text_pool  = (text  * m).sum(1) / m.sum(1)
        pooled     = torch.cat([audio_pool, text_pool], dim=-1)
        logits     = self.classifier(pooled)
        if return_pooled:
            return logits, pooled.detach()
        # return logits, self.domain_classifier(pooled, alpha)
        return logits


class ConGrAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.text_proj  = nn.Linear(cfg.text_dim,  cfg.d_model)
        self.co_attn = nn.ModuleList([
            CoAttentionBlock(cfg.d_model, cfg.ca_heads, cfg.dropout)
            for _ in range(cfg.rgat_layers)
        ])
        self.fusion    = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.rgat      = RGAT(cfg.d_model, cfg.rgat_layers, cfg.rgat_heads, cfg.dropout)
        self.pool_gate = nn.Linear(cfg.d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        # self.domain_classifier = DomainClassifier(cfg.d_model, cfg.n_domains, cfg.dropout)
        self.window    = cfg.graph_window
        self.drop_edge = cfg.drop_edge

    def forward(self, audio, text, mask, speakers, alpha=1.0, return_pooled=False):
        B, L    = audio.shape[:2]
        lengths = mask.sum(dim=1).long()

        audio = self.audio_proj(audio)
        text  = self.text_proj(text)
        for layer in self.co_attn:
            audio, text = layer(audio, text, ~mask)

        h = self.fusion(torch.cat([audio, text], dim=-1))
        h = h + sinusoidal_encoding(L, h.size(-1), h.device)

        x, edge_index, edge_type, batch, spk_mask = graphify(
            h, lengths, speakers, self.window, h.device, drop_edge=self.drop_edge
        )
        x = self.rgat(x, edge_index, edge_type)

        x_par = x[spk_mask]
        b_par = batch[spk_mask]
        att   = self.pool_gate(x_par).squeeze(-1)

        pooled = torch.zeros(B, x.size(-1), device=x.device)
        for b in range(B):
            m = (b_par == b)
            if m.any():
                w = torch.softmax(att[m], dim=0)
                pooled[b] = (w.unsqueeze(-1) * x_par[m]).sum(0)

        logits = self.classifier(pooled)
        if return_pooled:
            return logits, pooled.detach()
        # return logits, self.domain_classifier(pooled, alpha)
        return logits
