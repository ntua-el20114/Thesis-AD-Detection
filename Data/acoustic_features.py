""""
This script is written by Michael Raftopoulos, NTUA, Greece.
"""

import json
import sys
from pathlib import Path
import opensmile
from joblib import Parallel, delayed
import warnings


AUDIO_DIR = Path('_Combined/audio') 
JSON_FILES = [
    Path('_Combined/test_english.jsonl'),
    Path('_Combined/train_english.jsonl'),
]
NUM_CORES = 11 # For CPU parallelization

# Suppress multiprocessing warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def find_audio_file(json_entry):
    """This is here only for debugging purposes."""
    file_id = json_entry['File_ID']
    dataset = json_entry['Dataset']
    matches = []
    
    # Try both filename patterns and both extensions
    filename_patterns = [
        f"{dataset}_{file_id}", 
        f"{file_id}"             
    ]
    for pattern in filename_patterns:
        for ext in ['.wav', '.mp3']:
            file_path = AUDIO_DIR / f"{pattern}{ext}"
            if file_path.exists():
                matches.append(file_path)
    
    if matches:
        if len(matches) > 1:
            print(f"Warning: Multiple matches found for {file_id} from {dataset}: {matches}")
        return matches[0]
    return None


def extract_egemaps_features(audio_path):
    """Extract eGeMAPS features using openSMILE"""    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    # Extract features with timestamps
    features = smile.process_file(audio_path)
    
    return features.values.flatten().tolist()


def process_entry(entry):
    """
    Process a single entry and extract eGeMAPS features
    Returns new entry and success flag
    """
    file = find_audio_file(entry)
    if not file:
        return entry, False
    
    try:
        entry['egemaps'] = extract_egemaps_features(file)
        return entry, True
    except Exception as e:
        print(f"Error processing {entry.get('Dataset')}_{entry.get('File_ID')}: {str(e)}")
        return entry, False


def main():
    entries_found = 0
    successful_operations = 0

    for json_path in JSON_FILES:
        if not json_path.exists():
            print(f"JSON file not found: {json_path}")
            continue

        # Load JSONL data
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line in {json_path}: {e}")
                    continue

        found_count = 0

        # Extract features in parallel using multiple cores
        print(f"Loaded {len(data)} entries from {json_path.name}. Using {NUM_CORES} cores for processing...")
        
        results = Parallel(n_jobs=NUM_CORES, backend='threading', verbose=10)(
            delayed(process_entry)(entry) for entry in data
        )
        
        # Update entries and count successes
        data = []
        for entry, success in results:
            data.append(entry)
            if success:
                found_count += 1
        
        print(f"Found {found_count} audio files in {json_path.name}\n")

        # Update open JSONL file with new features
        with open(json_path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Saved updated entries to {json_path}\n")

        successful_operations += found_count
        entries_found += len(data)

    print(f"Total entries processed: {entries_found}")
    print(f"Total successful operations: {successful_operations}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean exit to suppress multiprocessing warnings
        sys.exit(0)
