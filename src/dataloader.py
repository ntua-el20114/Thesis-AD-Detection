import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

TRAIN_VAL_SPLIT_SEED = 42

# Define which keys are required for each model type to avoid redundant I/O
MODEL_REQUIREMENTS = {
    'wavlm': ['audio'],
    'test_linear': ['egemaps', 'bert'],
}

class MultiConAD_Dataset(Dataset):
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_dir: Union[str, Path],
        sample_rate: int = 16000,
        model_name: str = 'test_linear',
    ):
        self.jsonl_path = Path(jsonl_path)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.model_name = model_name
        
        # Determine requirements; default to audio if model unknown
        self.required_features = MODEL_REQUIREMENTS.get(model_name, ['audio'])
        
        self.records = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        output = {**record}
        
        # Load Audio
        if 'audio' in self.required_features:
            audio_filename = record['Audio_file']
            if audio_filename.startswith('Audio/'):
                audio_filename = audio_filename[6:]
            audio_path = self.audio_dir / audio_filename
            
            waveform, sr = torchaudio.load(str(audio_path))
            
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            output['audio'] = waveform
            
        # Load Features
        if 'egemaps' in self.required_features:
            output['egemaps'] = record['egemaps']
            
        if 'bert' in self.required_features:
            output['bert'] = record['bert']
            
        return output


def collate_fn_pad(batch: List[Dict]) -> Dict:
    """
    Dynamic collate function that only processes keys present in the batch.
    """
    if not batch:
        return {}
        
    keys = batch[0].keys()
    output = {}
    
    # Handle Audio (Padding)
    if 'audio' in keys:
        max_audio_length = max(sample['audio'].shape[1] for sample in batch)
        audio_batch = []
        audio_lengths = []
        
        for sample in batch:
            audio = sample['audio'].squeeze(0)
            audio_len = audio.shape[0]
            audio_lengths.append(audio_len)
            
            if audio_len < max_audio_length:
                audio = torch.nn.functional.pad(audio, (0, max_audio_length - audio_len))
            audio_batch.append(audio)
            
        output['audio'] = torch.stack(audio_batch)
        output['audio_lengths'] = torch.tensor(audio_lengths, dtype=torch.long)

    # Handle numeric features (Stacking)
    if 'egemaps' in keys:
        output['egemaps'] = torch.stack([torch.as_tensor(s['egemaps'], dtype=torch.float32) for s in batch])
        
    if 'bert' in keys:
        output['bert'] = torch.stack([torch.as_tensor(s['bert'], dtype=torch.float32) for s in batch])

    # Handle Metadata (Pass through as lists)
    # Exclude the tensor keys
    processed_keys = {'audio', 'egemaps', 'bert'}
    for key in keys:
        if key not in processed_keys:
            output[key] = [sample[key] for sample in batch]
    
    return output


def create_dataloaders(
    train_jsonl: Union[str, Path],
    test_jsonl: Union[str, Path],
    audio_dir: Union[str, Path],
    model_name: str,
    batch_size: int = 32,
    sample_rate: int = 16000,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
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
    
    # Create datasets with explicit instantiation
    train_dataset = MultiConAD_Dataset(train_jsonl, audio_dir, sample_rate, model_name)
    train_dataset.records = train_records_split
    
    val_dataset = MultiConAD_Dataset(train_jsonl, audio_dir, sample_rate, model_name)
    val_dataset.records = val_records_split
    
    test_dataset = MultiConAD_Dataset(test_jsonl, audio_dir, sample_rate, model_name)
    
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