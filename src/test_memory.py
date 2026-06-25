import torch
import gc
from model import CoAttentionClassifier, ConGrAD

class DummyCfg:
    audio_dim = 128
    text_dim = 128
    d_model = 64
    ca_heads = 4
    ca_layers = 2
    dropout = 0.1
    n_classes = 2
    rgat_layers = 2
    rgat_heads = 4
    graph_window = 3
    drop_edge = 0.0

def test_memory():
    cfg = DummyCfg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    B = 4
    L = 500
    
    audio = torch.randn(B, L, cfg.audio_dim, device=device)
    text = torch.randn(B, L, cfg.text_dim, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    speakers = torch.randint(0, 2, (B, L), device=device)
    
    # Test CoAttentionClassifier
    model1 = CoAttentionClassifier(cfg).to(device)
    torch.cuda.reset_peak_memory_stats()
    out1 = model1(audio, text, mask)
    loss1 = out1.sum()
    loss1.backward()
    print("CoAttentionClassifier Peak Memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    
    del model1, out1, loss1
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test ConGrAD
    model2 = ConGrAD(cfg).to(device)
    torch.cuda.reset_peak_memory_stats()
    out2 = model2(audio, text, mask, speakers)
    loss2 = out2.sum()
    loss2.backward()
    print("ConGrAD Peak Memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

if __name__ == '__main__':
    test_memory()
