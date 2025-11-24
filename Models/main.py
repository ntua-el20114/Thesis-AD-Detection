import torch
import torch.nn as nn
from pathlib import Path
from dataloader import create_dataloader
from testmodels import SimpleFusionModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# Configuration
CONFIG = {
    'train_data': 'train_english.jsonl',
    'test_data': 'test_english.jsonl',
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 0.001,
    'hidden_dim': 256,
    'num_classes': 3,
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Get features
        egemaps = batch['egemaps'].to(device)
        bert = batch['bert'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(egemaps, bert)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get features
            egemaps = batch['egemaps'].to(device)
            bert = batch['bert'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(egemaps, bert)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, train_dataset = create_dataloader(
        CONFIG['train_data'], 
        batch_size=CONFIG['batch_size'], 
        shuffle=True
    )
    test_loader, test_dataset = create_dataloader(
        CONFIG['test_data'], 
        batch_size=CONFIG['batch_size'], 
        shuffle=False
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Initialize model
    print("Initializing model...")
    model = SimpleFusionModel(
        egemaps_dim=train_dataset.egemaps_dim,
        bert_dim=train_dataset.bert_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=CONFIG['num_classes']
    )
    model = model.to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    best_f1 = 0
    results = {
        'train': [],
        'test': []
    }
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch [{epoch + 1}/{CONFIG['epochs']}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Evaluate
        eval_metrics = evaluate(model, test_loader, criterion, device)
        print(f"  Test Loss: {eval_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Test Precision: {eval_metrics['precision']:.4f}")
        print(f"  Test Recall: {eval_metrics['recall']:.4f}")
        print(f"  Test F1-Score: {eval_metrics['f1']:.4f}")
        
        # Store results
        results['train'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc
        })
        results['test'].append({
            'epoch': epoch + 1,
            'loss': eval_metrics['loss'],
            'accuracy': eval_metrics['accuracy'],
            'precision': eval_metrics['precision'],
            'recall': eval_metrics['recall'],
            'f1': eval_metrics['f1']
        })
        
        # Save best model
        if eval_metrics['f1'] > best_f1:
            best_f1 = eval_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print("  ✓ Best model saved!")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to training_results.json")
    
    # Final evaluation with best model
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\nFinal Test Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()