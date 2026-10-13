from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced.csv"

df = pd.read_csv(DATA_PATH)

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    
    text = re.sub(r"<[^>]+>", " ", text)

    
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = re.sub(r"_+", " ", text)

   
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stop_words(text):
    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(filtered_words)



sample = df.iloc[0]

for column in TEXT_COLUMNS:
    original_text = sample[column]

    cleaned = clean_text(original_text)
    filtered = remove_stop_words(cleaned)

    print(f"\n--- {column} ---")
    print("ORİJİNAL:")
    print(str(original_text)[:400])

    print("\nTEMİZLENMİŞ:")
    print(cleaned[:400])

    print("\nSTOP WORDS SONRASI:")
    print(filtered[:400])

    print("-" * 60)