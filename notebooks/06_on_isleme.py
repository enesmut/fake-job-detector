# -*- coding: utf-8 -*-

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Proje klasörleri
PROJE_KOKU = Path(__file__).resolve().parent.parent
VERI_YOLU = PROJE_KOKU / "data" / "raw" / "balanced.csv"
CIKTI_KLASORU = PROJE_KOKU / "data" / "processed"

CIKTI_KLASORU.mkdir(parents=True, exist_ok=True)

# Veri setini yükle
df = pd.read_csv(VERI_YOLU)

METIN_SUTUNLARI = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def temiz_metin(metin):
    if pd.isna(metin):
        return ""

    metin = str(metin).lower()
    metin = re.sub(r"<[^>]+>", " ", metin)
    metin = re.sub(r"[^\w\s]", " ", metin, flags=re.UNICODE)
    metin = re.sub(r"_+", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()

    return metin


def durdurma_kelimelerini_kaldir(metin):
    kelimeler = metin.split()
    filtrelenmis = [
        kelime
        for kelime in kelimeler
        if kelime not in ENGLISH_STOP_WORDS
    ]
    return " ".join(filtrelenmis)


# Tüm ilanları temizle
for sutun in METIN_SUTUNLARI:
    df[sutun] = df[sutun].apply(temiz_metin)
    df[sutun] = df[sutun].apply(durdurma_kelimelerini_kaldir)

# Kullanılacak metin alanlarını birleştir
df["clean_text"] = df[METIN_SUTUNLARI].agg(" ".join, axis=1)
df["clean_text"] = df["clean_text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

# Boş metin ve etiket kontrolleri
print("Toplam ilan:", len(df))
print("Empty cleaned text:", (df["clean_text"] == "").sum())
print("\nSınıf dağılımı:")
print(df["fraudulent"].value_counts().sort_index())

# Temizlenmiş veri setini kaydet
cikti_yolu = CIKTI_KLASORU / "balanced_cleaned.csv"
df.to_csv(cikti_yolu, index=False, encoding="utf-8")

print("\nİlk ilan örneği:")
print(df[["clean_text", "fraudulent"]].head(1).to_string(index=False))
print("\nDosya kaydedildi:", cikti_yolu)
