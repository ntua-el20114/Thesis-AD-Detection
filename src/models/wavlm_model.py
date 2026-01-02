import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        
        if freeze_backbone:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, num_classes)
    
    def forward(self, audio, audio_lengths):
        # audio: (batch_size, seq_len)
        # audio_lengths: (batch_size,)
        
        # Create attention mask
        batch_size, seq_len = audio.shape
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=audio.device)
        for i, length in enumerate(audio_lengths):
            attention_mask[i, :length] = 1
        
        # WavLM expects raw audio
        outputs = self.wavlm(audio, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Mean pooling over sequence length
        pooled = last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
        
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        return logits
