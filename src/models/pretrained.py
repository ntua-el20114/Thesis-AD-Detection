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
        
        # Download TRILLsson model (kagglehub will cache it automatically)
        print(f"Loading model from {model_source}...")
        model_path = kagglehub.model_download(model_source, force_download=False)
        print(f"Model loaded from: {model_path}")
        
        # Store the model path for cleanup
        self.model_path = model_path
        
        # Load the TensorFlow model
        self.trill_model = tf.saved_model.load(model_path)
        self.trill_infer = self.trill_model.signatures['serving_default']
        
        # Get the feature dimension from TRILLsson
        # TRILLsson outputs 512-dimensional features by default
        self.feature_dim = 1024
        
        # Classifier head
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES)
    
    def __del__(self):
        return
        """ Model destructor. Cleans up model files."""
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
        return

    
    def forward(self, audio, audio_lengths):
        """
        audio: (batch_size, seq_len) - raw audio waveform
        audio_lengths: (batch_size,) - lengths of each audio sample
        """
        
        batch_size, seq_len = audio.shape
        chunk_size = 32000  # 2 seconds at 16kHz

        # Convert PyTorch tensor to TensorFlow tensor
        # Need to move to CPU first for numpy conversion, then to TF tensor on GPU
        audio = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        
        with tf.device('/GPU:0'):  # Explicitly place on GPU
            audio = tf.constant(audio, dtype=tf.float32)

            # Split audio into 2-second chunks
            num_chunks = seq_len // chunk_size
            
            # Reshape to (batch_size, num_chunks, chunk_size)
            audio_chunks = tf.reshape(audio[:, :num_chunks * chunk_size], 
                                     [batch_size, num_chunks, chunk_size])
            del audio
            
            # Process each chunk through TRILLsson
            # We need to reshape to (batch_size * num_chunks, chunk_size) for processing
            audio_chunks = tf.reshape(audio_chunks, [batch_size * num_chunks, chunk_size])
            
            # Get embeddings for all chunks
            trill_embeddings = self.trill_infer(audio_chunks)
            
            # Reshape back to (batch_size, num_chunks, feature_dim)
            trill_embeddings = tf.reshape(trill_embeddings['tf.math.reduce_mean'], 
                                         [batch_size, num_chunks, self.feature_dim])
            
            # Mean pooling across chunks dimension
            pooled_embeddings = tf.reduce_mean(trill_embeddings, axis=1)
        
        # Convert back to PyTorch tensor for classification
        pooled_embeddings_pt = torch.from_numpy(pooled_embeddings.numpy())
        
        # Move to classifier device
        pooled_embeddings_pt = pooled_embeddings_pt.to(next(self.classifier.parameters()).device)
        
        # Classifier
        logits = self.classifier(pooled_embeddings_pt)  # (batch_size, num_classes)
        
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


