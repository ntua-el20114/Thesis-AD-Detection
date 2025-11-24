import numpy as np
# from openai import OpenAI
import whisper
import opensmile
import librosa
import pandas as pd
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import urllib.request
import os

# 1. OpenSMILE eGeMAPS Feature Extraction
def extract_egemaps_features(audio_path):
    """Extract eGeMAPS features using openSMILE"""
    print("\n=== Step 1: Extracting eGeMAPS Features ===")
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    # Extract features with timestamps
    features = smile.process_file(audio_path)
    
    print(f"Extracted {len(features.columns)} acoustic features")
    print(f"Feature shape: {features.shape}")
    print(f"Start Time: {features.index.min()}s, End Time: {features.index.max()}s")
    print(f"Sample features: {list(features.columns[:5])}")
    
    return features


# 2. Whisper Speech-to-Text with Word Timestamps (Local)
def transcribe_with_whisper_local(audio_path):
    """Transcribe audio using local Whisper model and get word-level timestamps"""
    print(f"\n=== Step 2: Whisper Transcription (Local) ===")
    
    # Load the model (will download if not already present)
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", or "large"
    
    # Transcribe the audio
    print("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        language="English", 
        verbose=True    # Show progress
    )
    
    print(f"Transcription: {result['text']}")
    if 'language' in result:
        print(f"Language: {result['language']}")
    
    # Extract word-level timestamps from segments
    words_with_timestamps = []
    for segment in result['segments']:
        # Add word with the segment's timestamp
        words_with_timestamps.append({
            'word': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end']
        })
    
    print(f"\nExtracted {len(words_with_timestamps)} segments with timestamps")
    if words_with_timestamps:
        print("Sample segments:", words_with_timestamps[:3])
    
    return result, words_with_timestamps


# 3. Acoustic-Lingual Alignment
def align_acoustic_linguistic(egemaps_features, words_with_timestamps):
    """Align acoustic features with linguistic units (words)"""
    print("\n=== Step 3: Acoustic-Lingual Alignment ===")
    
    aligned_data = []
    
    for word_info in words_with_timestamps:
        word = word_info['word']
        start_time = word_info['start']
        end_time = word_info['end']
        
        # Find acoustic features within this time window
        mask = (egemaps_features.index >= start_time) & (egemaps_features.index < end_time)
        word_features = egemaps_features[mask]
        
        if len(word_features) > 0:
            # Aggregate features for this word (mean, std, etc.)
            feature_stats = {
                'word': word,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'num_frames': len(word_features)
            }
            
            # Add mean values of key acoustic features
            for col in word_features.columns[:10]:  # First 10 features
                feature_stats[f'{col}_mean'] = word_features[col].mean()
                feature_stats[f'{col}_std'] = word_features[col].std()
            
            aligned_data.append(feature_stats)
    
    aligned_df = pd.DataFrame(aligned_data)
    
    print(f"Created alignment for {len(aligned_df)} words")
    print(f"\nAlignment DataFrame shape: {aligned_df.shape}")
    print("\nSample alignment:")
    print(aligned_df[['word', 'start_time', 'end_time', 'duration', 'num_frames']].head())
    
    return aligned_df


# 4. Visualization
def visualize_alignment(egemaps_features, words_with_timestamps, aligned_df):
    """Visualize the acoustic-lingual alignment"""
    print("\n=== Step 4: Visualization ===")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Acoustic features over time
    ax1 = axes[0]
    feature_to_plot = egemaps_features.columns[0]  # First feature
    ax1.plot(egemaps_features.index, egemaps_features[feature_to_plot], 
             linewidth=0.5, alpha=0.7)
    ax1.set_ylabel(f'{feature_to_plot}')
    ax1.set_title('Acoustic Features Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add word boundaries
    for word_info in words_with_timestamps:
        ax1.axvline(word_info['start'], color='red', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Plot 2: Word timeline
    ax2 = axes[1]
    for i, word_info in enumerate(words_with_timestamps):
        start = word_info['start']
        duration = word_info['end'] - word_info['start']
        ax2.barh(0, duration, left=start, height=0.5, alpha=0.7)
        ax2.text(start + duration/2, 0, word_info['word'], 
                ha='center', va='center', fontsize=8, rotation=0)
    
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel('Words')
    ax2.set_title('Word-Level Segmentation')
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Plot 3: Feature statistics per word
    ax3 = axes[2]
    if len(aligned_df) > 0:
        feature_col = [col for col in aligned_df.columns if '_mean' in col][0]
        word_centers = [(row['start_time'] + row['end_time']) / 2 
                       for _, row in aligned_df.iterrows()]
        ax3.scatter(word_centers, aligned_df[feature_col], alpha=0.7, s=100)
        ax3.set_ylabel(f'{feature_col}')
        ax3.set_title('Aligned Feature Statistics per Word')
        ax3.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('acoustic_lingual_alignment.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: acoustic_lingual_alignment.png")
    plt.show()


# Main experiment pipeline
def run_experiment():
    """Run the complete acoustic-lingual alignment experiment"""
    print("="*60)
    print("ACOUSTIC-LINGUAL ALIGNMENT EXPERIMENT")
    print("="*60)

    # Use existing audio file
    audio_path = "OSR_us_000_0010_8k.wav"
    
    # Extract eGeMAPS features
    egemaps_features = extract_egemaps_features(audio_path)
    
    # Transcribe with Whisper (Local)
    transcription, words_with_timestamps = transcribe_with_whisper_local(audio_path)
    
    # Align acoustic and linguistic features
    aligned_df = align_acoustic_linguistic(egemaps_features, words_with_timestamps)
    
    # Visualize results
    visualize_alignment(egemaps_features, words_with_timestamps, aligned_df)
    
    # Save aligned data
    aligned_df.to_csv('aligned_acoustic_linguistic_features.csv', index=False)
    print(f"\nSaved aligned features to: aligned_acoustic_linguistic_features.csv")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    
    return egemaps_features, words_with_timestamps, aligned_df


if __name__ == "__main__":
    # Run the experiment
    egemaps, words, aligned = run_experiment()
    
    # Additional analysis suggestions
    print("\n📊 Next Steps for Your Project:")
    print("1. Replace sample audio with your own speech recordings")
    print("2. Analyze correlations between acoustic features and linguistic units")
    print("3. Build ML models using aligned features (e.g., emotion recognition)")
    print("4. Process multiple audio files in batch")
    print("5. Try different openSMILE feature sets (ComParE, emobase)")
    print("\n💡 Note: The OpenAI Whisper API provides high-quality transcriptions")
    print("   and costs approximately $0.006 per minute of audio.")