import torch
from torch.utils.data import Dataset, DataLoader
import json


class SpeechDataset(Dataset):
    """Dataset for loading egemaps and BERT features from JSONL file"""
    
    def __init__(self, jsonl_path, label_mapping=None):
        """
        Args:
            jsonl_path: Path to the JSONL file
            label_mapping: Dictionary mapping diagnosis strings to integer labels
                          If None, will create automatically from unique diagnoses
        """
        self.data = []
        
        # Load data from JSONL file
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Filter out 'Unknown' diagnoses
                if entry.get('Diagnosis', 'Unknown') != 'Unknown':
                    self.data.append(entry)
        
        # Create label mapping if not provided
        if label_mapping is None:
            unique_diagnoses = sorted(set(entry.get('Diagnosis', 'Unknown') for entry in self.data))
            self.label_mapping = {diag: idx for idx, diag in enumerate(unique_diagnoses)}
        else:
            self.label_mapping = label_mapping
        
        self.num_classes = len(self.label_mapping)
        
        # Verify feature dimensions from first entry
        if self.data:
            self.egemaps_dim = len(self.data[0]['egemaps'])
            self.bert_dim = len(self.data[0]['bert'])
            print(f"Loaded {len(self.data)} samples")
            print(f"egemaps dimension: {self.egemaps_dim}")
            print(f"BERT dimension: {self.bert_dim}")
            print(f"Label mapping: {self.label_mapping}")
            print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Convert features to tensors
        egemaps = torch.tensor(entry['egemaps'], dtype=torch.float32)
        bert = torch.tensor(entry['bert'], dtype=torch.float32)
        
        # Get label from diagnosis
        diagnosis = entry.get('Diagnosis', 'Unknown')
        label = self.label_mapping.get(diagnosis, 0)  # Default to 0 if not found
        label = torch.tensor(label, dtype=torch.long)
        
        # You can also return metadata if needed
        metadata = {
            'pid': entry.get('PID', ''),
            'diagnosis': diagnosis,
            'age': entry.get('Age', -1),
            'gender': entry.get('Gender', '')
        }
        
        return {
            'egemaps': egemaps,
            'bert': bert,
            'label': label,
            'metadata': metadata
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle metadata dictionaries
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Dictionary with batched tensors and metadata list
    """
    egemaps = torch.stack([item['egemaps'] for item in batch])
    bert = torch.stack([item['bert'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]  # Keep as list
    
    return {
        'egemaps': egemaps,
        'bert': bert,
        'label': labels,
        'metadata': metadata
    }


def create_dataloader(jsonl_path, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader from JSONL file
    
    Args:
        jsonl_path: Path to JSONL file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        dataloader: PyTorch DataLoader
        dataset: The dataset object (useful for getting dimensions)
    """
    dataset = SpeechDataset(jsonl_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # Use custom collate function
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, dataset
