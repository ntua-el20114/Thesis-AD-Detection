import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, save_results_csv, extract_metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define mapping to match utils.TARGET_NAMES: ['HC', 'MCI', 'Dementia']
LABEL_MAP = {'HC': 0, 'MCI': 1, 'Dementia': 2}

def load_data_from_jsonls(file_paths):
    texts = []
    labels = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = data.get("Text_interviewer_participant", "")
                        label_str = data.get("Diagnosis", "")
                        
                        if text and label_str in LABEL_MAP:
                            texts.append(text)
                            labels.append(LABEL_MAP[label_str])
                    except json.JSONDecodeError:
                        continue
    return texts, labels

def main():
    parser = argparse.ArgumentParser(description="Extract TF-IDF from jsonls and classify using SVM/RF.")
    parser.add_argument("--train_files", nargs='+', default=["../../data/MultiConAD/train_English.jsonl"], 
                        help="Specific jsonl files for training")
    parser.add_argument("--test_files", nargs='+', default=["../../data/MultiConAD/test_English.jsonl"], 
                        help="Specific jsonl files for testing")
    parser.add_argument("--base_seed", type=int, default=24, help="Initial base seed for experiments")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to repeat the experiment")
    
    args = parser.parse_args()

    print(f"Loading training data from {args.train_files}...")
    X_train, y_train = load_data_from_jsonls(args.train_files)
    
    print(f"Loading testing data from {args.test_files}...")
    X_test, y_test = load_data_from_jsonls(args.test_files)
    
    if not X_train or not X_test:
        print("No data loaded. Please check the provided paths.")
        return

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("\nExtracting TF-IDF features...")
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    svm_metrics = []
    rf_metrics = []

    for i in range(args.runs):
        seed = args.base_seed + i
        print(f"\n--- Run {i+1}/{args.runs} (Seed: {seed}) ---")
        
        # Initialize everything with the current seed
        set_seed(seed)
        
        # SVM
        svm_clf = SVC(kernel='linear', random_state=seed)
        svm_clf.fit(X_train_tfidf, y_train)
        svm_preds = svm_clf.predict(X_test_tfidf)
        
        svm_res = extract_metrics(y_test, svm_preds)
        svm_metrics.append(svm_res)
        print(f"SVM Accuracy: {svm_res['accuracy']:.4f}, UAR: {svm_res['UAR']:.4f}")

        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_clf.fit(X_train_tfidf, y_train)
        rf_preds = rf_clf.predict(X_test_tfidf)
        
        rf_res = extract_metrics(y_test, rf_preds)
        rf_metrics.append(rf_res)
        print(f"RF Accuracy: {rf_res['accuracy']:.4f}, UAR: {rf_res['UAR']:.4f}")

    # Save results to CSVs in the baseline directory
    out_dir = Path(__file__).parent
    
    svm_csv_path = out_dir / "svm_results.csv"
    save_results_csv(svm_metrics, svm_csv_path, args.base_seed)
    print(f"\nSVM results saved to {svm_csv_path}")

    rf_csv_path = out_dir / "rf_results.csv"
    save_results_csv(rf_metrics, rf_csv_path, args.base_seed)
    print(f"RF results saved to {rf_csv_path}")

if __name__ == "__main__":
    main()
