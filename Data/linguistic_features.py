from pathlib import Path
import json
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# Install required packages (run this in Colab)
# !pip install transformers torch tqdm

JSON_FILES = [
    # Path('_Combined/test_english.jsonl'),
    # Path('_Combined/train_english.jsonl'),
    Path('_Combined/debug.jsonl'),
]

def get_bert_embedding(text, tokenizer, model, device):
    """Extract BERT embedding for given text."""
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # Get BERT output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding (first token) as sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding.cpu().numpy().tolist()

def process_jsonl_file(json_path, tokenizer, model, device):
    """Process a JSONL file and add BERT embeddings."""
    print(f"\nProcessing: {json_path}")
    
    # Read all entries
    entries = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    # Process each entry
    updated_entries = []
    for entry in tqdm(entries, desc=f"Extracting embeddings from {json_path.name}"):
        text = entry.get('Text_interviewer_participant', '')
        
        if text:
            # Get BERT embedding
            embedding = get_bert_embedding(text, tokenizer, model, device)
            entry['bert'] = embedding
        else:
            entry['bert'] = None
        
        updated_entries.append(entry)
    
    # Write back to original file
    with open(json_path, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✓ Completed: {json_path} ({len(updated_entries)} entries)")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BERT model and tokenizer
    print("\nLoading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    print("✓ BERT model loaded")
    
    # Process each file
    for json_path in JSON_FILES:
        if json_path.exists():
            process_jsonl_file(json_path, tokenizer, model, device)
        else:
            print(f"⚠ File not found: {json_path}")
    
    print("\n✓ All files processed successfully!")

if __name__ == "__main__":
    main()