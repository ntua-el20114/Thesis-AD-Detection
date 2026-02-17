import shutil
import os

import torch
import torch.nn as nn
from transformers import WavLMModel
import tensorflow as tf
import kagglehub
import numpy as np

NUM_CLASSES = 3

class TRILLssonClassifier(nn.Module):
    """
    Uses the TRILLsson5 model, an AST with 88.6M parameters.
    The model is only available through TensorFlow, so we need to perform an awkward PyTorch/TensorFlow merging
    """

    def __init__(self, freeze_backbone: bool = False, model_source: str = "google/trillsson/tensorFlow2/5"):
        super().__init__()
        
        # Download TRILLsson model 
        print(f"Downloading model from {model_source}...")
        model_path = kagglehub.model_download(model_source)
        print(f"Downloaded model stored in {model_path}")
        
        # Store the model path for cleanup
        self.model_path = model_path
        
        # Load the TensorFlow model
        self.trill_model = tf.saved_model.load(model_path)
        
        # Get the feature dimension from TRILLsson
        # TRILLsson outputs 512-dimensional features by default
        self.feature_dim = 512
        
        # Classifier head
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES)
    
    def __del__(self):
        """ Model destructor. Cleans up model files."""
        return
        try:
            if hasattr(self, 'model_path') and self.model_path:
                # Remove the entire model directory
                if os.path.exists(self.model_path):
                    shutil.rmtree(self.model_path)
                    print(f"Cleaned up TRILLsson model files from: {self.model_path}")
                
                # Also try to remove parent directory if it's empty
                parent_dir = os.path.dirname(self.model_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    shutil.rmtree(parent_dir)
                    print(f"Cleaned up empty parent directory: {parent_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up model files: {e}")
    
    def forward(self, audio, audio_lengths):
        """
        audio: (batch_size, seq_len) - raw audio waveform
        audio_lengths: (batch_size,) - lengths of each audio sample
        """
        
        batch_size, seq_len = audio.shape
        
        # Convert PyTorch tensor to numpy for TensorFlow
        audio_np = audio.detach().cpu().numpy()
        
        # Process each sample in the batch
        batch_features = []
        for i in range(batch_size):
            # Get individual sample
            sample = audio_np[i, :audio_lengths[i]]
            
            # TRILLsson expects audio at 16kHz, mono, float32
            # Reshape to (samples,) if needed
            if len(sample.shape) > 1:
                sample = np.mean(sample, axis=-1)  # Convert to mono if stereo
            
            # Normalize audio to [-1, 1] range
            sample = sample.astype(np.float32)
            if np.max(np.abs(sample)) > 0:
                sample = sample / np.max(np.abs(sample))
            
            # Add batch dimension for TRILLsson
            sample = np.expand_dims(sample, axis=0)
            
            # Get TRILLsson features
            # TRILLsson returns a dictionary with 'embedding' and 'logits'
            features = self.trill_model(sample)
            
            # Use the embedding features
            if isinstance(features, dict) and 'embedding' in features:
                embedding = features['embedding'].numpy()
            else:
                # If it's not a dict, assume it's the embedding directly
                embedding = features.numpy()
            
            # Mean pooling over time
            pooled_features = np.mean(embedding, axis=0)  # (feature_dim,)
            batch_features.append(pooled_features)
        
        # Convert back to PyTorch tensor
        batch_features = torch.from_numpy(np.stack(batch_features)).to(audio.device)
        
        # Classifier
        logits = self.classifier(batch_features)  # (batch_size, num_classes)
        
        return logits

class WavLMClassifier(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        
        if freeze_backbone:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, NUM_CLASSES)
    
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


