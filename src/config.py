from dataclasses import dataclass, fields
from pathlib import Path
import yaml

@dataclass
class Config:
    # Paths
    gemma_dir:       str
    trill_dir:       str
    speakers_json:   str
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
    n_classes:     int
    d_model:       int
    ca_heads:      int
    ca_layers:     int
    rgat_heads:    int
    rgat_layers:   int
    graph_window:  int
    skip_log_base: int

    # Regularization
    dropout:        float
    drop_edge:      float
    weight_decay:   float
    balanced_mixup: bool
    mixup_alpha:    float

    # Training
    device:     str
    batch_size: int
    lr:         float
    epochs:     int
    patience:   int

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
