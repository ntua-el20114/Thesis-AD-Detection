import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}

def _convert_diagnosis_to_labels(diagnosis_list, device):
    """Efficiently convert diagnosis strings to label tensor."""
    return torch.tensor([LABEL_MAP.get(d, 0) for d in diagnosis_list], dtype=torch.long, device=device)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch tensors to device
        batch['audio'] = batch['audio'].to(device)
        batch['audio_lengths'] = batch['audio_lengths'].to(device)
        batch['egemaps'] = batch['egemaps'].to(device)
        batch['bert'] = batch['bert'].to(device)
        
        # Convert diagnosis to labels (HC=0, MCI=1, Dementia=2)
        labels = _convert_diagnosis_to_labels(batch['Diagnosis'], device)
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch tensors to device
            batch['audio'] = batch['audio'].to(device)
            batch['audio_lengths'] = batch['audio_lengths'].to(device)
            batch['egemaps'] = batch['egemaps'].to(device)
            batch['bert'] = batch['bert'].to(device)
            
            labels = _convert_diagnosis_to_labels(batch['Diagnosis'], device)
            
            logits = model(batch)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train a model.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
    
    Returns:
        Tuple of (model, history dictionary, test metrics dictionary)
    """
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    print(f"Model moved to device: {device}")
    print(f"Model device check: {next(model.parameters()).device}")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # Evaluate on test set at the end
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    test_metrics = {'test_loss': test_loss, 'test_accuracy': test_acc}
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*50 + "\n")
    
    return model, history, test_metrics
