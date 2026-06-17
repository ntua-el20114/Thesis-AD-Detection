import json, torch
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence


LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}

# Known MultiConAD/DementiaBank corpus prefixes → display names
_CORPUS_PREFIXES = {
    'pitt':      'Pitt',
    'adress':    'ADReSS',
    'taukadial': 'TAUKADIAL',
    'tauk':      'TAUKADIAL',
    'delaware':  'Delaware',
    'wls':       'WLS',
    'kempler':   'Kempler',
    'vas':       'VAS',
    'baycrest':  'Baycrest',
    'lu':        'Lu',
}

# Corpus → integer domain ID.
# n_domains in config.yaml must equal the number of entries present in your data.
DOMAIN_MAP = {
    'Pitt':      0,
    'ADReSS':    1,
    'TAUKADIAL': 2,
    'Delaware':  3,
    'WLS':       4,
    'Kempler':   5,
    'VAS':       6,
    'Baycrest':  7,
    'Lu':        8,
}


def extract_corpus(file_name: str) -> str:
    """Infer source corpus from file name prefix (e.g. 'Pitt_001_PAR' -> 'Pitt')."""
    prefix = file_name.split('_')[0].lower()
    for key, name in _CORPUS_PREFIXES.items():
        if prefix.startswith(key):
            return name
    return file_name.split('_')[0]   # fallback: use prefix verbatim


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class MultiConADDataset(Dataset):
    def __init__(self, jsonl_path, gemma_dir, trill_dir, speakers_path):
        self.gemma_dir = Path(gemma_dir)
        self.trill_dir = Path(trill_dir)
        self.speakers  = json.loads(Path(speakers_path).read_text())
        all_samples    = load_jsonl(jsonl_path)
        self.samples   = [
            s for s in all_samples
            if (self.gemma_dir / f"{s['File_Name']}.pt").exists()
            and (self.trill_dir / f"{s['File_Name']}.pt").exists()
            and s['File_Name'] in self.speakers
        ]
        # Corpus name and domain ID per sample
        self.corpus_names = [extract_corpus(s['File_Name']) for s in self.samples]
        self.domain_ids   = [DOMAIN_MAP.get(c, 0) for c in self.corpus_names]
        print(f"Loaded {len(self.samples)}/{len(all_samples)} samples with embeddings")

    def __getitem__(self, idx):
        s         = self.samples[idx]
        name      = s['File_Name']
        trill     = torch.load(self.trill_dir / f"{name}.pt")
        gemma     = torch.load(self.gemma_dir / f"{name}.pt")
        speakers  = torch.tensor(self.speakers[name],   dtype=torch.long)
        label     = torch.tensor(LABEL_MAP[s['Diagnosis']], dtype=torch.long)
        domain_id = torch.tensor(self.domain_ids[idx],  dtype=torch.long)
        return trill, gemma, speakers, label, domain_id

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    trills, gemmas, speakers, labels, domain_ids = zip(*batch)
    lengths      = torch.tensor([t.size(0) for t in trills])
    trill_pad    = pad_sequence(trills,   batch_first=True)
    gemma_pad    = pad_sequence(gemmas,   batch_first=True)
    speakers_pad = pad_sequence(speakers, batch_first=True, padding_value=0)
    mask         = torch.arange(trill_pad.size(1)) < lengths.unsqueeze(1)
    return trill_pad, gemma_pad, speakers_pad, mask, torch.stack(labels), torch.stack(domain_ids)


def make_balanced_sampler(dataset: MultiConADDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that gives each diagnosis class equal expected draw rate,
    oversampling MCI and Dementia to match the HC majority.
    """
    labels        = torch.tensor([LABEL_MAP[s['Diagnosis']] for s in dataset.samples])
    class_counts  = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
