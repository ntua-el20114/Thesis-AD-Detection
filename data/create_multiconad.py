import os
import sys
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
import random

# --- Configuration ---
ROOT_DIR = Path(".").resolve()
MULTICONAD_DIR = ROOT_DIR / "MultiConAD"
AUDIO_OUTPUT_DIR = MULTICONAD_DIR / "Audio"

# Dataset Directories that need fixing
DATASETS = ["WLS", "Pitt", "Lu", "Kempler", "Delaware", "Baycrest", "VAS"]

# --- Step 0: Fix Directory Structures ---
def fix_directory_names():
    """
    MultiConAD's scripts hardcode the folder name 'Transcriptions'.
    Your file structure uses 'Transcripts'.
    This function renames them automatically.
    """
    print("\n--- [0/5] Normalizing Directory Names ---")
    for ds in DATASETS:
        ds_path = ROOT_DIR / ds
        if not ds_path.exists(): continue
        
        current = ds_path / "Transcripts"
        target = ds_path / "Transcriptions"
        
        # If 'Transcripts' exists but 'Transcriptions' doesn't, rename/move it
        if current.exists() and not target.exists():
            print(f"  Renaming: {current} -> {target}")
            try:
                shutil.move(str(current), str(target))
            except Exception as e:
                print(f"  [!] Failed to move {current}: {e}")

# --- Helper: Audio Linking ---
def find_audio(base_folder, filename_stem):
    """Finds audio file recursively (mp3/wav) in base_folder."""
    for ext in ['*.mp3', '*.wav', '*.m4a']:
        matches = list(Path(base_folder).rglob(f"{filename_stem}{ext}"))
        if matches:
            return matches[0]
    return None

def create_symlink(src, link_name):
    """Creates a symlink in MultiConAD/Audio."""
    if not src or not src.exists(): return None
    dst = AUDIO_OUTPUT_DIR / link_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
        return link_name
    except OSError:
        return None

# --- Step 1: Pre-Process WLS (Generate wls.jsonl) ---
def preprocess_wls():
    print("\n--- [1/5] Pre-Processing WLS ---")
    wls_dir = ROOT_DIR / "WLS"
    
    # 1. Locate Metadata
    candidates = [
        wls_dir / "WLS-data.xlsx",
        ROOT_DIR / "WLS-data.xlsx",
        wls_dir / "WLS-data.xlsx - Below normal fluency - 2011.csv"
    ]
    meta_path = next((p for p in candidates if p.exists()), None)
    
    if not meta_path:
        print(f"  [!] Skipping WLS: Metadata not found.")
        return

    print(f"  Loading Metadata: {meta_path.name}")
    try:
        if str(meta_path).endswith('.csv'):
            df = pd.read_csv(meta_path)
        else:
            df = pd.read_excel(meta_path, sheet_name='Below normal fluency - 2011')
    except Exception as e:
        print(f"  [!] Error reading metadata: {e}")
        return

    # Find flag column
    flag_col = next((c for c in df.columns if "1 sd below" in c), None)
    
    # Debug: Print columns if flag not found
    if not flag_col:
        print("  [!] '1 sd below' column not found.")
        print("  Available columns:", df.columns.tolist())
        return

    # 2. Process Files
    records = []
    # Search in both 'Transcriptions' (new name) and root WLS
    cha_files = list(wls_dir.rglob("*.cha"))
    
    if not cha_files:
        print("  [!] No .cha files found in WLS.")
        return

    print(f"  Found {len(cha_files)} .cha files. Matching IDs...")

    for cha in cha_files:
        stem = cha.stem # e.g. "00007"
        
        # Robust ID Matching
        # Metadata IDs: 2000000007. Filename: 00007.
        # Logic: Meta ID ends with filename stem (as int or string)
        match = None
        
        # Heuristic: convert stem to int to remove leading zeros, compare suffixes
        try:
            stem_int = int(stem) # 00007 -> 7
            suffix_str = str(stem_int) # "7"
        except:
            suffix_str = stem

        for _, row in df.iterrows():
            if pd.isna(row['idtlkbnk']): continue
            
            meta_id_str = str(int(row['idtlkbnk']))
            
            # Check if meta_id ends with the stem (e.g. 2000000007 ends with 00007 or 7)
            if meta_id_str.endswith(stem) or meta_id_str.endswith(suffix_str):
                match = row
                break
        
        if match is None:
            # Uncomment to debug specific misses
            # print(f"  No match for file: {stem}")
            continue

        # Label Logic
        flag = str(match[flag_col]).strip().upper()
        if flag == 'Y': label = "AD"
        elif flag == 'N': label = "HC"
        else: continue

        # Parse Text
        try:
            with open(cha, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l.replace("*PAR:", "").strip() for l in f if l.startswith("*PAR:")]
            text = " ".join(lines)
        except: text = ""
        
        if not text: continue

        records.append({
            "text": text,
            "label": label,
            "id": f"WLS_{stem}",
            "original_id": str(match['idtlkbnk']),
            "dataset": "WLS",
            "meta": {"age": match.get('age 2011'), "education": match.get('education')}
        })

    out_file = wls_dir / "wls.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  -> Generated {out_file} ({len(records)} records)")

# --- Step 2: Pre-Process Taukdial ---
def preprocess_taukdial():
    print("\n--- [2/5] Pre-Processing Taukdial ---")
    tauk_dir = ROOT_DIR / "Taukdial"
    meta_path = tauk_dir / "testgroundtruth.csv"
    trans_path = ROOT_DIR / "Tauk-Transcribe" / "transcripts.json"

    if not meta_path.exists() or not trans_path.exists():
        print(f"  [!] Skipping Taukdial (Missing csv or transcripts.json).")
        return

    # Load Metadata
    try:
        df = pd.read_csv(meta_path)
    except:
        print("  [!] Error reading Taukdial CSV.")
        return
    
    # Load Transcripts
    with open(trans_path, 'r') as f:
        trans_data = json.load(f)
    
    # Build Map
    trans_map = {}
    if isinstance(trans_data, list):
        for item in trans_data:
            path = item.get('path', '')
            stem = Path(path).stem
            trans_map[stem] = item.get('text', '')
    else:
        for path, text in trans_data.items():
            trans_map[Path(path).stem] = text

    records = []
    # Identify ID column (could be 'session', 'id', 'subject')
    id_col = next((c for c in df.columns if c.lower() in ['session', 'id', 'subject']), None)
    if not id_col:
        print("  [!] Taukdial ID column not found.")
        return

    for _, row in df.iterrows():
        sid = str(row[id_col])
        if not sid or sid not in trans_map: continue
        
        # Label
        diag = str(row.get('diagnosis') or row.get('label', '')).lower()
        if 'ad' in diag: label = 'AD'
        elif 'mci' in diag: label = 'MCI'
        else: label = 'HC'

        records.append({
            "text": trans_map[sid],
            "label": label,
            "id": f"TAUK_{sid}",
            "dataset": "Taukdial",
            "meta": {}
        })

    out_file = tauk_dir / "taukdial.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  -> Generated {out_file} ({len(records)} records)")

# --- Step 3: Run MultiConAD Scripts ---
def run_pipeline_scripts():
    print("\n--- [3/5] Running MultiConAD Scripts ---")
    
    # Ensure collection.py can find the files we just made
    # collection.py typically glob searches for .jsonl
    
    scripts = ["cha_collection.py", "collection.py", "text_cleaning_English.py"]
    
    for script in scripts:
        script_path = ROOT_DIR / script
        if script_path.exists():
            print(f"  Running {script}...")
            # We run with cwd=ROOT_DIR to ensure relative paths work
            res = subprocess.run([sys.executable, script_path.name], cwd=ROOT_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [!] Error in {script}:\n{res.stderr}")
                # Don't abort immediately, try to proceed
            else:
                print(f"  {script} done.")
        else:
            print(f"  [!] Warning: {script} not found.")

# --- Step 4: Post-Process ---
def post_process():
    print("\n--- [4/5] Audio Linking & Splitting ---")
    
    input_file = MULTICONAD_DIR / "combined_English.jsonl"
    if not input_file.exists():
        print("  [!] combined_English.jsonl missing. Creating fallback from individual JSONLs...")
        # Fallback: Merge all jsonls we found
        all_recs = []
        for j in ROOT_DIR.rglob("*.jsonl"):
            if "MultiConAD" in str(j): continue
            with open(j, 'r') as f:
                for line in f:
                    try: all_recs.append(json.loads(line))
                    except: pass
        if not all_recs:
            print("  [!] No data found at all.")
            return
        # Write temporary combined
        with open(input_file, 'w') as f:
            for r in all_recs: f.write(json.dumps(r)+"\n")

    if not AUDIO_OUTPUT_DIR.exists():
        AUDIO_OUTPUT_DIR.mkdir(parents=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_records = []
    
    # Dataset Mapping for Audio Search
    folder_map = {
        "Pitt": "Pitt", "Lu": "Lu", "WLS": "WLS", 
        "Taukdial": "Taukdial", "Delaware": "Delaware", 
        "Kempler": "Kempler", "Baycrest": "Baycrest", "VAS": "VAS"
    }

    for line in lines:
        try: rec = json.loads(line)
        except: continue
        
        ds = rec.get('dataset', '')
        rec_id = rec.get('id', '')
        
        # Extract stem (Dataset_ID -> ID)
        if "_" in rec_id:
            stem = rec_id.split('_', 1)[1]
        else:
            stem = rec_id
        
        # Special logic for Taukdial stems if needed
        # For WLS, stems are "00007"
        
        link_name = None
        if ds in folder_map:
            base = ROOT_DIR / folder_map[ds]
            audio_src = find_audio(base, stem)
            if audio_src:
                link_name = f"{rec_id}{audio_src.suffix}"
                create_symlink(audio_src, link_name)
        
        rec['file_name'] = link_name
        final_records.append(rec)

    # Splitting
    train, test = [], []
    random.seed(42)
    
    from collections import defaultdict
    groups = defaultdict(list)
    for r in final_records:
        groups[r.get('dataset', 'Unknown')].append(r)

    print("  Splitting Data:")
    for ds, items in groups.items():
        if ds == 'WLS':
            train.extend(items)
            print(f"    {ds}: All {len(items)} -> Train")
        else:
            random.shuffle(items)
            cut = int(len(items) * 0.8)
            train.extend(items[:cut])
            test.extend(items[cut:])
            print(f"    {ds}: {len(items[:cut])} Train, {len(items[cut:])} Test")

    # Save Final
    def save(n, d):
        with open(MULTICONAD_DIR / n, 'w') as f:
            for i in d: f.write(json.dumps(i)+"\n")

    save("train_english.jsonl", train)
    save("test_english.jsonl", test)
    save("combined_English.jsonl", train + test)
    
    print(f"\n--- [5/5] Pipeline Complete ---")
    print(f"  Train: {len(train)}")
    print(f"  Test:  {len(test)}")
    print(f"  Audio: {AUDIO_OUTPUT_DIR}")

if __name__ == "__main__":
    if not MULTICONAD_DIR.exists():
        MULTICONAD_DIR.mkdir()
        
    fix_directory_names()
    preprocess_wls()
    preprocess_taukdial()
    run_pipeline_scripts()
    post_process()
