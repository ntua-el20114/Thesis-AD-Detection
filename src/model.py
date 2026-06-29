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
        self.fusion = nn.Linear(2*cfg.d_model, cfg.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        # self.domain_classifier = DomainClassifier(2 * cfg.d_model, cfg.n_domains, cfg.dropout)

    def forward(self, audio, text, mask, speakers=None, alpha=1.0, return_pooled=False):
        # speakers unused; accepted for interface compatibility with ConGrAD
        kpm = ~mask
        audio = self.audio_proj(audio)
        text = self.text_proj(text)
        for layer in self.layers:
            audio, text = layer(audio, text, kpm)
        m = mask.unsqueeze(-1).float()
        audio_pool = (audio * m).sum(1) / m.sum(1)
        text_pool = (text  * m).sum(1) / m.sum(1)
        # pooled = torch.cat([audio_pool, text_pool], dim=-1)
        pooled = self.fusion(torch.cat([audio_pool, text_pool], dim=-1))
        logits = self.classifier(pooled)
        if return_pooled:
            return logits, pooled.detach()
        # return logits, self.domain_classifier(pooled, alpha)
        return logits


class ConGrAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.text_proj = nn.Linear(cfg.text_dim,  cfg.d_model)
        self.co_attn = nn.ModuleList([
            CoAttentionBlock(cfg.d_model, cfg.ca_heads, cfg.dropout)
            for _ in range(cfg.rgat_layers)
        ])
        self.fusion = nn.Linear(2*cfg.d_model, cfg.d_model)
        
        # Calculate number of relations for Relative Positional Embedding
        D = 2 * cfg.graph_window + 2
        n_relations = 4 * D
        
        self.rgat = RGAT(cfg.d_model, cfg.rgat_layers, cfg.rgat_heads, cfg.dropout, n_relations=n_relations)
        self.att_score = nn.Linear(2*cfg.d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2*cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        # self.domain_classifier = DomainClassifier(cfg.d_model, cfg.n_domains, cfg.dropout)
        self.window = cfg.graph_window
        self.drop_edge = cfg.drop_edge

    def forward(self, audio, text, mask, speakers, alpha=1.0, return_pooled=False):
        B, L = audio.shape[:2]
        lengths = mask.sum(dim=1).long()

        # Project modalities
        audio = self.audio_proj(audio)
        text = self.text_proj(text)

        # Multimodal Fusion
        for layer in self.co_attn:
            audio, text = layer(audio, text, ~mask)
        h = self.fusion(torch.cat([audio, text], dim=-1))

        # Graph Construction (using Relative Positional Embeddings in edge_type)
        h_flat, edge_index, edge_type, batch, spk_mask = graphify(
            h, lengths, speakers, self.window, h.device, drop_edge=self.drop_edge
        )

        # RGAT
        x = self.rgat(h_flat, edge_index, edge_type)

        # Concatenate pre-rgat features
        x = torch.cat([h_flat, x], dim=-1)

        # x_par = x[spk_mask]
        # b_par = batch[spk_mask]
        x_par = x
        b_par = batch
        att = self.att_score(x_par).squeeze(-1)

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

