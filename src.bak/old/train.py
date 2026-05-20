import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, recall_score
from typing import Union

LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}

def _convert_diagnosis_to_labels(diagnosis_list, device):
    """Efficiently convert diagnosis strings to label tensor."""
    return torch.tensor([LABEL_MAP.get(d, 0) for d in diagnosis_list], dtype=torch.long, device=device)

def prepare_batch(batch, device):
    """
    Generic function to prepare batch for the model:
    1. Moves all Tensor values to the target device.
    2. Separates model inputs (Tensors) from metadata (lists/strings).
    """
    model_inputs = {}
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # Move to device and add to inputs
            model_inputs[k] = v.to(device)
    
    return model_inputs

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Extract only the tensors (audio, egemaps, bert, etc.) and move to GPU
        inputs = prepare_batch(batch, device)
        
        # Prepare Labels
        labels = _convert_diagnosis_to_labels(batch['Diagnosis'], device)
        
        # Forward Pass
        # **inputs unpacks the dictionary into arguments
        optimizer.zero_grad()
        logits = model(**inputs) 
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = prepare_batch(batch, device)
            labels = _convert_diagnosis_to_labels(batch['Diagnosis'], device)
            
            logits = model(**inputs)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    uar = recall_score(all_labels, all_preds, average='macro')
    return {
        'loss': avg_loss,
        'acc': accuracy,
        'f1': f1,
        'uar': uar
    }



def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    device = torch.device(device)
    
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)
        val_loss = metrics['loss']
        val_acc = metrics['acc']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test UAR: {test_metrics['uar']:.4f}")
    print("="*50 + "\n")
    
    return model, history, test_metrics
