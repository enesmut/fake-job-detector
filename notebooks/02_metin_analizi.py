from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Proje ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Veriyi yükle
df = pd.read_csv(DATA_PATH)

# İncelenecek metin alanları
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# Eksik değerleri analiz sırasında boş metin olarak ele al.
for column in text_columns:
    df[column] = df[column].fillna("").astype(str)

# Her ilanın açıklama uzunluğunu hesapla.
df["description_length"] = df["description"].str.len()

print("\n--- SINIFLARA GÖRE AÇIKLAMA UZUNLUĞU ---")

print(
    df.groupby("fraudulent")["description_length"]
    .agg(["count", "mean", "median", "min", "max"])
)

# Açıklama uzunluklarının karşılaştırılması
plt.figure(figsize=(9, 5))

for label, name in [(0, "Gerçek"), (1, "Sahte")]:
    values = df.loc[
        df["fraudulent"] == label,
        "description_length"
    ]

    plt.hist(
        values,
        bins=40,
        alpha=0.5,
        label=name
    )

plt.title("Gerçek ve Sahte İlanların Açıklama Uzunlukları")
plt.xlabel("Karakter sayısı")
plt.ylabel("İlan sayısı")
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "aciklama_uzunluklari.png",
    dpi=150
)

plt.close()

# Metin örneklerini incele
print("\n--- GERÇEK İLAN ÖRNEKLERİ ---")

print(
    df.loc[
        df["fraudulent"] == 0,
        ["title", "description"]
    ].head(3).to_string(index=False)
)

print("\n--- SAHTE İLAN ÖRNEKLERİ ---")

print(
    df.loc[
        df["fraudulent"] == 1,
        ["title", "description"]
    ].head(3).to_string(index=False)
)

# Metinlerde HTML kalıntılarını kontrol et
print("\n--- HTML KALINTISI KONTROLÜ ---")

for column in text_columns:
    html_count = df[column].str.contains(
        r"<[^>]+>",
        regex=True,
        na=False
    ).sum()

    print(f"{column}: {html_count} kayıtta HTML benzeri ifade")

print("\nAnaliz tamamlandı.")
print("Grafik reports/aciklama_uzunluklari.png konumuna kaydedildi.")