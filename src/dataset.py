import json, torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class MultiConADDataset(Dataset):
    def __init__(self, jsonl_path, gemma_dir, trill_dir):
        self.gemma_dir = Path(gemma_dir)
        self.trill_dir = Path(trill_dir)
        all_samples = load_jsonl(jsonl_path)
        self.samples = [
            s for s in all_samples
            if (self.gemma_dir / f"{s['File_Name']}.pt").exists()
            and (self.trill_dir / f"{s['File_Name']}.pt").exists()
        ]
        print(f"Loaded {len(self.samples)}/{len(all_samples)} samples with embeddings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        name = s['File_Name']
        trill = torch.load(self.trill_dir / f"{name}.pt")  # [N, 1024]
        gemma = torch.load(self.gemma_dir / f"{name}.pt")  # [N, 786]
        label = torch.tensor(LABEL_MAP[s['Diagnosis']], dtype=torch.long)
        return trill, gemma, label


def collate_fn(batch):
    trills, gemmas, labels = zip(*batch)
    lengths  = torch.tensor([t.size(0) for t in trills])
    trill_pad = pad_sequence(trills, batch_first=True)  # [B, max_N, 1024]
    gemma_pad = pad_sequence(gemmas, batch_first=True)  # [B, max_N, 786]
    mask = torch.arange(trill_pad.size(1)) < lengths.unsqueeze(1)  # [B, max_N], True=valid
    return trill_pad, gemma_pad, mask, torch.stack(labels)
