from src.dataloader import create_dataloaders
from pathlib import Path

# Paths to your data
data_dir = Path("data/MultiConAD")
train_jsonl = data_dir / "train_English.jsonl"
test_jsonl = data_dir / "test_English.jsonl"
audio_dir = data_dir / "Audio"

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    train_jsonl=train_jsonl,
    test_jsonl=test_jsonl,
    audio_dir=audio_dir,
    batch_size=4,
    sample_rate=16000,
)

# Test train loader
print("Testing train loader...")
batch = next(iter(train_loader))
print(f"Audio shape: {batch['audio'].shape}")
print(f"Audio lengths: {batch['audio_lengths']}")
print(f"Diagnosis (sample): {batch['Diagnosis'][:2]}")
print(f"Text length (sample): {batch['Text_length'][:2]}")
print(f"Keys in batch: {list(batch.keys())}")

# Test test loader
print("\nTesting test loader...")
batch = next(iter(test_loader))
print(f"Audio shape: {batch['audio'].shape}")
print(f"Batch size: {len(batch['Diagnosis'])}")
print("✓ Dataloader works!")