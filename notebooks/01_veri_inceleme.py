from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Veri seti bulunamadı: {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print("\n--- VERİ SETİ BİLGİLERİ ---")
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")

print("\n--- SÜTUNLAR ---")
print(df.columns.tolist())

print("\n--- İLK 5 KAYIT ---")
print(df.head())

print("\n--- VERİ TÜRLERİ ---")
print(df.dtypes)

print("\n--- EKSİK DEĞER ANALİZİ ---")

missing = pd.DataFrame({
    "Eksik Sayısı": df.isnull().sum(),
    "Eksik Yüzdesi": (
        df.isnull().mean() * 100
    ).round(2)
})

print(
    missing.sort_values(
        by="Eksik Yüzdesi",
        ascending=False
    )
)

print("\n--- SINIF DAĞILIMI ---")
print(
    df["fraudulent"]
    .value_counts()
    .sort_index()
)

print("\n--- TEKRARLANAN SATIRLAR ---")
print(df.duplicated().sum())

print("\n--- METİN ALANLARI ---")

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

for column in text_columns:
    text = df[column].fillna("").astype(str).str.strip()

    empty_count = (text == "").sum()

    print(
        f"{column}: {empty_count} boş kayıt "
        f"(%{empty_count / len(df) * 100:.2f})"
    )

print("\nVeri incelemesi tamamlandı.")