import os, torch, json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("/home/michraf/Thesis/data/SeamlessInteraction/naturalistic/train")
OUTPUT = Path("/home/michraf/Thesis/data/SeamlessInteraction/gemma")

def extract_gemma():
    print(f"\n{'='*40}\nExtracting Gemma (Merged Interactions)\n{'='*40}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-300M").to(device)

    interactions = defaultdict(list)
    for json_path in DATA_DIR.rglob("*.json"):
        interaction_id = json_path.stem.rsplit('_', 1)[0]
        interactions[interaction_id].append(json_path)

    for interaction_id, json_paths in tqdm(interactions.items(), desc="Processing"):
        out_pt = OUTPUT / f"{interaction_id}.pt"
        if out_pt.exists(): 
            continue
        
        all_turns = []
        for json_path in json_paths:
            with open(json_path) as f:
                data = json.load(f)
            
            for turn in data.get("metadata:transcript", []):
                text = turn.get("transcript", "").strip()
                if text:
                    all_turns.append({
                        "start": float(turn.get("start", 0)),
                        "text": text
                    })
        
        if not all_turns: continue
        
        # Chronological sort
        all_turns.sort(key=lambda x: x["start"])
        texts = [t["text"] for t in all_turns]
        
        embeddings = torch.tensor(model.encode(texts), dtype=torch.float32)
        torch.save(embeddings, out_pt)

    del model

if __name__ == "__main__":
    extract_gemma()
