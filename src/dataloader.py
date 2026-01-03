import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


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
        
        # torchaudio.load() handles MP3, WAV, FLAC, etc.
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
    
    for sample in batch:
        audio = sample['audio'].squeeze(0)
        audio_len = audio.shape[0]
        audio_lengths.append(audio_len)
        
        if audio_len < max_audio_length:
            audio = torch.nn.functional.pad(audio, (0, max_audio_length - audio_len))
        
        audio_batch.append(audio)
    
    output = {
        'audio': torch.stack(audio_batch),
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
    }
    
    for key in batch[0].keys():
        if key != 'audio':
            output[key] = [sample[key] for sample in batch]
    
    return output


def create_dataloaders(
    train_jsonl: Union[str, Path],
    val_jsonl: Union[str, Path],
    test_jsonl: Union[str, Path],
    audio_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    sample_rate: int = 16000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MultiConAD_Dataset(train_jsonl, audio_dir, sample_rate)
    val_dataset = MultiConAD_Dataset(val_jsonl, audio_dir, sample_rate)
    test_dataset = MultiConAD_Dataset(test_jsonl, audio_dir, sample_rate)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
    )
    
    return train_loader, val_loader, test_loader
