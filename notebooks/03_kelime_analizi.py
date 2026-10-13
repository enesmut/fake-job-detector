from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt

# Proje yolları
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Veri setini yükle
df = pd.read_csv(DATA_PATH)

# İncelenecek metin alanları
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# Eksik değerleri boş metin olarak ele al
for column in text_columns:
    df[column] = df[column].fillna("").astype(str)

# Kelimeleri küçük harfe çevirerek çıkar
def extract_words(text):
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


# Gerçek ve sahte ilanların kelimelerini birleştir
for label, label_name in [(0, "Gerçek"), (1, "Sahte")]:
    subset = df[df["fraudulent"] == label]

    all_text = " ".join(
        subset[text_columns].agg(" ".join, axis=1)
    )

    words = extract_words(all_text)

    # En sık geçen 20 kelime
    most_common = Counter(words).most_common(20)

    result = pd.DataFrame(
        most_common,
        columns=["Kelime", "Frekans"]
    )

    print(f"\n--- {label_name.upper()} İLANLARDA EN SIK 20 KELİME ---")
    print(result.to_string(index=False))

    # Sonuçları CSV olarak kaydet
    result.to_csv(
        REPORTS_PATH / f"en_sik_kelimeler_{label_name.lower()}.csv",
        index=False
    )

    # Grafik oluştur
    plt.figure(figsize=(10, 6))

    plt.barh(
        result["Kelime"][::-1],
        result["Frekans"][::-1]
    )

    plt.title(f"{label_name} İlanlarda En Sık 20 Kelime")
    plt.xlabel("Frekans")
    plt.ylabel("Kelime")
    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / f"kelime_frekans_{label_name.lower()}.png",
        dpi=150
    )

    plt.close()

print("\nKelime analizi tamamlandı.")