import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/home/michraf/Thesis/data/SeamlessInteraction/naturalistic/train")
SPK_FILE = Path("/home/michraf/Thesis/data/SeamlessInteraction/speakers.json")
JSONL_OUT = Path("/home/michraf/Thesis/data/SeamlessInteraction/train_seamless.jsonl")

def structure():
    speakers_dict = {}
    existing_ids = set()
    
    if SPK_FILE.exists():
        with open(SPK_FILE, "r") as f:
            speakers_dict = json.load(f)
            existing_ids = set(speakers_dict.keys())
            print(f"Loaded {len(existing_ids)} existing interactions.")

    interactions = defaultdict(list)
    for json_path in DATA_DIR.rglob("*.json"):
        interaction_id = json_path.stem.rsplit('_', 1)[0]
        if interaction_id not in existing_ids:
            interactions[interaction_id].append(json_path)

    if not interactions:
        print("No new interactions found. Exiting.")
        return

    new_count = 0
    jsonl_data = []
    
    for interaction_id, json_paths in interactions.items():
        all_turns = []
        for i, json_path in enumerate(sorted(json_paths)):
            spk_label = 1 if i == 0 else 0
            
            with open(json_path) as f:
                data = json.load(f)
                
            for turn in data.get("metadata:transcript", []):
                if turn.get("transcript", "").strip():
                    all_turns.append({
                        "start": float(turn.get("start", 0)),
                        "speaker": spk_label
                    })
                    
        if not all_turns: continue
        
        all_turns.sort(key=lambda x: x["start"])
        
        speakers_dict[interaction_id] = [t["speaker"] for t in all_turns]
        jsonl_data.append({"File_Name": interaction_id, "Diagnosis": "HC"})
        new_count += 1

    with open(SPK_FILE, "w") as f: 
        json.dump(speakers_dict, f)
        
    mode = "a" if JSONL_OUT.exists() else "w"
    with open(JSONL_OUT, mode) as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {new_count} new merged interactions (Total: {len(speakers_dict)}).")

if __name__ == "__main__":
    structure()
