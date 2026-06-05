import os, torch, librosa, kagglehub, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATA_DIR = Path("/home/michraf/Thesis/data/SeamlessInteraction/naturalistic/train")
OUTPUT = Path("/home/michraf/Thesis/data/SeamlessInteraction/trillsson")
SAMPLE_RATE = 24000

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth error: {e}")

def extract_trillsson():
    print(f"\n{'='*40}\nExtracting TRILLsson (Merged Interactions)\n{'='*40}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = kagglehub.model_download("google/trillsson/tensorFlow2/5")
    model = tf.saved_model.load(model_path).signatures['serving_default']
    
    # Group files by Interaction ID (e.g., V00_S0030_I00000126)
    interactions = defaultdict(list)
    for wav_path in DATA_DIR.rglob("*.wav"):
        interaction_id = wav_path.stem.rsplit('_', 1)[0]
        interactions[interaction_id].append(wav_path)

    for interaction_id, wav_paths in tqdm(interactions.items(), desc="Processing"):
        out_pt = OUTPUT / f"{interaction_id}.pt"
        if out_pt.exists(): 
            continue
        
        all_turns = []
        for wav_path in wav_paths:
            json_path = wav_path.with_suffix('.json')
            if not json_path.exists(): continue
            
            with open(json_path) as f:
                data = json.load(f)
                
            for turn in data.get("metadata:transcript", []):
                all_turns.append({
                    "start": float(turn.get("start", 0)),
                    "end": float(turn.get("end", float(turn.get("start", 0)) + 0.1)),
                    "wav_path": wav_path
                })
        
        if not all_turns: continue
        
        # Chronological sort
        all_turns.sort(key=lambda x: x["start"])
        
        embeddings = []
        for turn in all_turns:
            duration = max(0.1, turn["end"] - turn["start"])
            try:
                waveform, _ = librosa.load(turn["wav_path"], sr=SAMPLE_RATE, offset=turn["start"], duration=duration)
                audio_tf = tf.constant(waveform.astype(np.float32), dtype=tf.float32)
                with tf.device('/GPU:0'):
                    res = model(tf.expand_dims(audio_tf, 0))
                    emb = tf.reduce_mean(res['tf.math.reduce_mean'], axis=0)
                embeddings.append(torch.tensor(np.atleast_1d(emb.numpy()), dtype=torch.float32))
            except Exception:
                embeddings.append(torch.zeros(1024, dtype=torch.float32))
        
        if embeddings:
            torch.save(torch.stack(embeddings), out_pt)

    del model

if __name__ == "__main__":
    extract_trillsson()
