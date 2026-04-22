"""
This is a slightly modified version of the "text_cleaning_English.py" script
from the MultiConAD repo:
https://github.com/ArezoShakeri/MultiConAD

Besides the modified file paths, this script:
 - Ensures there are no WLS samples on the test set
 - Ensures there is no speaker leakage from the train set to the test set
 - Adds 'Patient_ID' and 'File_Name' fields
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner


# Pitt, Lu, Baycrest, VAS, Kempler, WLS, Delware, taukdial_English_train, taukdial_English_test
input_files = [
    "Pitt/pitt.jsonl",
    "Lu/lu.jsonl",
    "Baycrest/baycrest.jsonl",
    "VAS/vas.jsonl",
    "Kempler/kempler.jsonl",
    "WLS/wls.jsonl",
    "Delaware/delaware.jsonl",
    "Taukdial/taukdial.jsonl",
]
output_directory = 'MultiConAD/'
output_filename = 'English.jsonl'
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()
train_filename = 'train_' + output_filename
test_filename = 'test_' + output_filename

output_path = f"{output_directory}/{output_filename}"

English_df = pd.read_json(output_path, lines=True)

# Remove Chinese transcript from Taukdial
def remove_zh_language_rows(df):
    return df[df['Languages'] != 'zh']

def clean_diagnosis(df):
    # Remove specific diagnoses
    diagnoses_to_remove = ['Vascular', 'Memory', 'Aphasia', "Pick's", 'Other']
    df = df[~df['Diagnosis'].isin(diagnoses_to_remove)]
    # Remove rows with empty Diagnosis
    df = df[df['Diagnosis'].notna() & (df['Diagnosis'] != '')]
    
    # Rename diagnoses
    df['Diagnosis'] = df['Diagnosis'].replace({
        'Control': 'HC',
        'Conrol': 'HC',      # There is a single sample in Lu with the 'Conrol' typo
        'NC': 'HC',
        'H': 'HC',
        'AD': 'Dementia',
        'PossibleAD': 'Dementia',
        'ProbableAD': 'Dementia',
        'potential dementia': 'Dementia',
        'D': 'Dementia',
        "Alzheimer's": 'Dementia'
    })
    
    return df

def preprocess_text(text):
    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('PAR', '')
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\\x[0-9A-Za-z_]+\\x', '', text) 
    text = re.sub(r'\b\w+:\s*', '', text) 
    text = text.replace('\n', ' ')
    text = text.replace('→', '')
    text = text.replace('(', '').replace(')', '')
    text = re.sub(r'[\\+^"/„]', '', text)
    text = re.sub(r"[_']", '', text)
    text = text.replace('\t', ' ')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('&=laughs', '')
    text = text.replace('&=nods', '')
    text = text.replace('&=coughs', '')
    text = text.replace('&=snaps:tongue', '')
    text = text.replace('<', '').replace('>', '')
    text = text.replace('*', '').replace('&', '')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.,!?;:])\s+\1', r'\1', text)
    text = re.sub(r'(\.\s*){2,}', '.', text)
    if '.' in text:
        text = text.rsplit('.', 1)[0] + '.' 

    return text

def remove_short_transcripts(df, min_length=60):
    return df[df['Text_length'] > min_length]

def extract_patient_id(text):
    return re.sub(r'-(\d+)$', '', text)

English_df = remove_zh_language_rows(English_df)
English_df = clean_diagnosis(English_df)
English_df["Text_interviewer_participant"] = English_df["Text_interviewer_participant"].apply(preprocess_text)

English_df["File_Name"] = English_df["Dataset"] + "_" + English_df["File_ID"]
English_df["Patient_ID"] = English_df["File_Name"].apply(extract_patient_id)

English_df['Text_length'] = English_df['Text_interviewer_participant'].apply(len)

English_df = remove_short_transcripts(English_df)

# Exclude WLS samples from the test set
WLS_df = English_df[English_df['Dataset'] == 'WLS']
English_df = English_df[English_df['Dataset'] != 'WLS']

# Split patients
unique_patients = English_df['Patient_ID'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, stratify=English_df.groupby('Patient_ID')['Diagnosis'].first(), random_state=42)

# Create splits based on patients
train_en = English_df[English_df['Patient_ID'].isin(train_patients)]
test_en = English_df[English_df['Patient_ID'].isin(test_patients)]

train_en = pd.concat([train_en, WLS_df], ignore_index=True)

# Save train and test datasets as JSONL
train_en.to_json(output_directory + train_filename, orient="records", lines=True, force_ascii=False)
test_en.to_json(output_directory + test_filename, orient="records", lines=True, force_ascii=False)

