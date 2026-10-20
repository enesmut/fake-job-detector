

# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib


# --------------------------------------------------
# 1. DOSYA YOLLARI
# --------------------------------------------------

PROJE_KOKU = Path(__file__).resolve().parent.parent

VERI_YOLU = PROJE_KOKU / "data" / "processed" / "balanced_cleaned.csv"
CIKTI_KLASORU = PROJE_KOKU / "data" / "processed"

CIKTI_KLASORU.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. VERI SETINI YUKLE
# --------------------------------------------------

df = pd.read_csv(VERI_YOLU)

df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["clean_text"].str.strip() != ""].copy()

X = df["clean_text"]
y = df["fraudulent"].astype(int)

print("Toplam ilan sayisi:", len(df))


# --------------------------------------------------
# 3. EGITIM VE TEST VERILERINI AYIR
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Egitim ilan sayisi:", len(X_train))
print("Test ilan sayisi:", len(X_test))


# --------------------------------------------------
# 4. TF-IDF ILE METINLERI SAYISALLASTIR
# --------------------------------------------------

vektorleyici = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vektorleyici.fit_transform(X_train)
X_test_tfidf = vektorleyici.transform(X_test)


# --------------------------------------------------
# 5. SAYISAL MATRISLERI KAYDET
# --------------------------------------------------

save_npz(
    CIKTI_KLASORU / "X_train_tfidf.npz",
    X_train_tfidf
)

save_npz(
    CIKTI_KLASORU / "X_test_tfidf.npz",
    X_test_tfidf
)

joblib.dump(
    vektorleyici,
    CIKTI_KLASORU / "tfidf_vektorleyici.joblib"
)


# --------------------------------------------------
# 6. EGITIM-TEST AYRIMINI KAYDET
# --------------------------------------------------

bolunme = pd.DataFrame({
    "veri_satiri": list(X_train.index) + list(X_test.index),
    "bolum": ["egitim"] * len(X_train) + ["test"] * len(X_test),
    "fraudulent": list(y_train) + list(y_test)
})

bolunme.to_csv(
    CIKTI_KLASORU / "veri_bolunmesi.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 7. RAPOR ICIN OKUNAKLI TF-IDF TABLOSU OLUSTUR
# --------------------------------------------------

# Egitim verisinde ortalama TF-IDF agirligi en yuksek 20 kelime
terimler = vektorleyici.get_feature_names_out()
ortalama_agirlik = X_train_tfidf.mean(axis=0).A1

en_iyi_indeksler = ortalama_agirlik.argsort()[-20:][::-1]
secili_terimler = terimler[en_iyi_indeksler]

# Ilk 5 egitim ilaninin secilen 20 kelimeye ait agirliklari
ornek = pd.DataFrame(
    X_train_tfidf[:5, en_iyi_indeksler].toarray(),
    columns=secili_terimler
)

ornek.insert(0, "fraudulent", y_train.iloc[:5].to_numpy())
ornek.insert(0, "veri_satiri", X_train.index[:5].to_numpy())

# Standart CSV dosyasi
ornek.to_csv(
    CIKTI_KLASORU / "tfidf_ornek_matrisi.csv",
    index=False,
    encoding="utf-8-sig"
)

# Excel'de kolay acilmasi icin alternatif dosya
ornek.to_csv(
    CIKTI_KLASORU / "tfidf_ornek_matrisi_excel.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";"
)


# --------------------------------------------------
# 8. SONUCLARI OZET DOSYASINA YAZ
# --------------------------------------------------

ozet = f"""
TF-IDF HAZIRLIK SONUCLARI
-------------------------
Kullanilan ilan sayisi: {len(df)}
Egitim ilan sayisi: {len(X_train)}
Test ilan sayisi: {len(X_test)}

Egitim sinif dagilimi:
{y_train.value_counts().sort_index().to_string()}

Test sinif dagilimi:
{y_test.value_counts().sort_index().to_string()}

TF-IDF ozellik sayisi: {X_train_tfidf.shape[1]}
Egitim matrisi boyutu: {X_train_tfidf.shape}
Test matrisi boyutu: {X_test_tfidf.shape}

Model egitimi: YAPILMADI
"""

(CIKTI_KLASORU / "tfidf_ozet.txt").write_text(
    ozet,
    encoding="utf-8"
)


# --------------------------------------------------
# 9. SONUCLARI TERMINALDE GOSTER
# --------------------------------------------------

print(ozet)

print("Olusturulan ciktilar:")
print("- X_train_tfidf.npz")
print("- X_test_tfidf.npz")
print("- tfidf_vektorleyici.joblib")
print("- veri_bolunmesi.csv")
print("- tfidf_ornek_matrisi.csv")
print("- tfidf_ornek_matrisi_excel.csv")
print("- tfidf_ozet.txt")

print("\nIslem basariyla tamamlandi.")
