import torch
import torch.nn as nn


class LinearFusionModel(nn.Module):
    """Simple model that processes egemaps and BERT features"""
    
    def __init__(self, egemaps_dim=88, bert_dim=768, hidden_dim=256, num_classes=3):
        """
        Args:
            egemaps_dim: Dimension of egemaps features
            bert_dim: Dimension of BERT features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (e.g., for classification)
        """
        super(LinearFusionModel, self).__init__()
        
        # Separate pathways for each modality
        self.egemaps_encoder = nn.Sequential(
            nn.Linear(egemaps_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.bert_encoder = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        egemaps = data["egemaps"]
        bert = data["bert"]
        del data
        
        # Encode each modality
        egemaps_encoded = self.egemaps_encoder(egemaps)
        bert_encoded = self.bert_encoder(bert)
        
        # Concatenate features
        fused = torch.cat([egemaps_encoded, bert_encoded], dim=1)
        
        # Final classification
        output = self.fusion(fused)
        
        return output