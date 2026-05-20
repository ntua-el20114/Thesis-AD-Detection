from dataclasses import dataclass, fields
from pathlib import Path
import yaml

@dataclass
class Config:
    # Paths
    gemma_dir:       str
    trill_dir:       str
    train_jsonl:     str
    test_jsonl:      str
    results_dir:     str

    # Experiment
    experiment_name:        str
    experiment_description: str
    base_seed:              int
    n_runs:                 int

    # Embedding dims
    audio_dim: int
    text_dim:  int

    # Model
    n_classes: int
    d_model:   int
    n_heads:   int
    n_layers:  int
    dropout:   float
    patience:  int

    # Training
    device:     str
    batch_size: int
    lr:         float
    epochs:     int

    # Hyperparameter Optimization
    hpo_epochs: int
    hpo_trials: int

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path) as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if isinstance(value, str) and value.startswith('~'):
                data[key] = str(Path(value).expanduser())
        return cls(**data)
