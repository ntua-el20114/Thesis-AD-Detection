import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        audio = batch['audio'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        diagnosis = batch['Diagnosis']
        
        # Convert diagnosis to labels (HC=0, MCI=1, Dementia=2)
        label_map = {'HC': 0, 'MCI': 1, 'Dementia': 2}
        labels = torch.tensor([label_map.get(d, 0) for d in diagnosis]).to(device)
        
        optimizer.zero_grad()
        logits = model(audio, audio_lengths)
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
            audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            diagnosis = batch['Diagnosis']
            
            label_map = {'HC': 0, 'MCI': 1, 'Dementia': 2}
            labels = torch.tensor([label_map.get(d, 0) for d in diagnosis]).to(device)
            
            logits = model(audio, audio_lengths)
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
        test_loader: Test dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    
    return model, history
