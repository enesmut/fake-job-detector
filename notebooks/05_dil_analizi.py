from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced.csv"

df = pd.read_csv(DATA_PATH)

DetectorFactory.seed = 42


def detect_language(text):
    if pd.isna(text):
        return "empty"

    text = str(text).strip()

    if len(text) < 30:
        return "too_short"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


df["detected_language"] = df["description"].apply(
    detect_language
)

print("\n--- DİL DAĞILIMI ---")

print(
    df["detected_language"].value_counts(dropna=False)
)

print("\n--- SINIFA GÖRE DİL DAĞILIMI ---")

print(
    pd.crosstab(
        df["fraudulent"],
        df["detected_language"],
        normalize="index"
    ).round(3)
)

print("\n--- ALMANCA OLDUĞU TAHMİN EDİLEN ÖRNEKLER ---")

german_samples = df[
    df["detected_language"] == "de"
][["title", "description", "fraudulent"]]

print(german_samples.head(5).to_string(index=False))

print("\nDil analizi tamamlandı.")