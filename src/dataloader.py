import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

TRAIN_VAL_SPLIT_SEED = 42

class MultiConAD_Dataset(Dataset):
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_dir: Union[str, Path],
        sample_rate: int = 16000,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        
        self.records = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        
        audio_filename = record['Audio_file']
        # Strip 'Audio/' prefix if present (since audio_dir already points to Audio/)
        if audio_filename.startswith('Audio/'):
            audio_filename = audio_filename[6:]
        audio_path = self.audio_dir / audio_filename
        
        waveform, sr = torchaudio.load(str(audio_path))
        
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return {
            **record,
            'audio': waveform,
        }


def collate_fn_pad(batch: List[Dict]) -> Dict:
    max_audio_length = max(sample['audio'].shape[1] for sample in batch)
    
    audio_batch = []
    audio_lengths = []
    egemaps_batch = []
    bert_batch = []
    
    for sample in batch:
        # Process audio
        audio = sample['audio'].squeeze(0)
        audio_len = audio.shape[0]
        audio_lengths.append(audio_len)
        
        if audio_len < max_audio_length:
            audio = torch.nn.functional.pad(audio, (0, max_audio_length - audio_len))
        
        audio_batch.append(audio)
        
        # Process egemaps and bert (convert to tensors)
        egemaps_batch.append(torch.as_tensor(sample['egemaps'], dtype=torch.float32))
        bert_batch.append(torch.as_tensor(sample['bert'], dtype=torch.float32))
    
    output = {
        'audio': torch.stack(audio_batch),
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
        'egemaps': torch.stack(egemaps_batch),
        'bert': torch.stack(bert_batch),
    }
    
    # Add all other fields (as lists for metadata)
    for key in batch[0].keys():
        if key not in ['audio', 'egemaps', 'bert']:
            output[key] = [sample[key] for sample in batch]
    
    return output


def create_dataloaders(
    train_jsonl: Union[str, Path],
    test_jsonl: Union[str, Path],
    audio_dir: Union[str, Path],
    batch_size: int = 32,
    sample_rate: int = 16000,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    The validation set is extracted from the training data.
    
    Args:
        train_jsonl: Path to training JSONL file
        test_jsonl: Path to test JSONL file
        audio_dir: Path to audio directory
        batch_size: Batch size for dataloaders
        sample_rate: Sample rate for audio
        val_split: Fraction of training data to use for validation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load training records
    train_records = []
    with open(train_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    classes = [record['Diagnosis'] for record in train_records]
    train_records_split, val_records_split = train_test_split(
        train_records,
        test_size=val_split,
        stratify=classes,
        random_state=TRAIN_VAL_SPLIT_SEED,
    )
    
    # Create datasets
    train_dataset = MultiConAD_Dataset.__new__(MultiConAD_Dataset)
    train_dataset.jsonl_path = Path(train_jsonl)
    train_dataset.audio_dir = Path(audio_dir)
    train_dataset.sample_rate = sample_rate

    train_dataset.records = train_records_split
    
    val_dataset = MultiConAD_Dataset.__new__(MultiConAD_Dataset)
    val_dataset.jsonl_path = Path(train_jsonl)
    val_dataset.audio_dir = Path(audio_dir)
    val_dataset.sample_rate = sample_rate

    val_dataset.records = val_records_split
    
    test_dataset = MultiConAD_Dataset(test_jsonl, audio_dir, sample_rate)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_pad,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad,
    )
    
    return train_loader, val_loader, test_loader
