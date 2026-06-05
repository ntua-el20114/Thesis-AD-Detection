from pathlib import Path
from seamless_interaction.fs import SeamlessInteractionFS, DatasetConfig

BATCHES = 30
WORKERS = 8

def download_batches():
    target_dir = Path("~/Thesis/data/SeamlessInteraction").expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        label="naturalistic",
        split="train",
        local_dir=target_dir,
        num_workers=WORKERS
    )
    
    fs = SeamlessInteractionFS(config=config)
    
    for batch_idx in range(BATCHES):
        batch_dir = target_dir / "naturalistic" / "train" / f"{batch_idx:04d}"
        
        if batch_dir.exists() and any(batch_dir.iterdir()):
            print(f"Batch {batch_idx} already exists at {batch_dir}. Skipping download.")
            continue
            
        print(f"Downloading naturalistic train batch {batch_idx}...")
        fs.download_batch_from_hf(batch_idx=batch_idx)

if __name__ == "__main__":
    download_batches()
