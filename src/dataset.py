import json, torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class MultiConADDataset(Dataset):
    def __init__(self, jsonl_path, gemma_dir, trill_dir, speakers_path):
        self.gemma_dir = Path(gemma_dir)
        self.trill_dir = Path(trill_dir)
        self.speakers = json.loads(Path(speakers_path).read_text())
        all_samples = load_jsonl(jsonl_path)
        self.samples = [
            s for s in all_samples
            if (self.gemma_dir / f"{s['File_Name']}.pt").exists()
            and (self.trill_dir / f"{s['File_Name']}.pt").exists()
            and s['File_Name'] in self.speakers
        ]
        print(f"Loaded {len(self.samples)}/{len(all_samples)} samples with embeddings")

    def __getitem__(self, idx):
        s = self.samples[idx]
        name = s['File_Name']
        trill = torch.load(self.trill_dir / f"{name}.pt")
        gemma = torch.load(self.gemma_dir / f"{name}.pt")
        speakers = torch.tensor(self.speakers[name], dtype=torch.long)
        label = torch.tensor(LABEL_MAP[s['Diagnosis']], dtype=torch.long)
        return trill, gemma, speakers, label

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    trills, gemmas, speakers, labels = zip(*batch)
    lengths      = torch.tensor([t.size(0) for t in trills])
    trill_pad    = pad_sequence(trills,   batch_first=True)
    gemma_pad    = pad_sequence(gemmas,   batch_first=True)
    speakers_pad = pad_sequence(speakers, batch_first=True, padding_value=0)
    mask = torch.arange(trill_pad.size(1)) < lengths.unsqueeze(1)
    return trill_pad, gemma_pad, speakers_pad, mask, torch.stack(labels)
