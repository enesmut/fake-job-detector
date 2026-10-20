
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. DOSYA YOLLARI
# ==================================================

PROJE_KLASORU = Path(__file__).resolve().parent.parent
VERI_KLASORU = PROJE_KLASORU / "data" / "processed"
SONUC_KLASORU = PROJE_KLASORU / "reports" / "model_results"

SONUC_KLASORU.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. VERILERI YUKLE
# ==================================================

X_train = load_npz(VERI_KLASORU / "X_train_tfidf.npz")
X_test = load_npz(VERI_KLASORU / "X_test_tfidf.npz")

bolunme = pd.read_csv(
    VERI_KLASORU / "veri_bolunmesi.csv"
)

y_train = bolunme.loc[
    bolunme["bolum"] == "egitim", "fraudulent"
].astype(int).to_numpy()

y_test = bolunme.loc[
    bolunme["bolum"] == "test", "fraudulent"
].astype(int).to_numpy()


# ==================================================
# 3. VERI KONTROLU
# ==================================================

assert X_train.shape[0] == len(y_train), (
    "Egitim verisi ile etiket sayisi uyusmuyor."
)

assert X_test.shape[0] == len(y_test), (
    "Test verisi ile etiket sayisi uyusmuyor."
)


# ==================================================
# 4. MODELI EGIT
# ==================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_skor = model.predict_proba(X_test)[:, 1]


# ==================================================
# 5. PERFORMANS METRIKLERI
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
)

roc_auc = roc_auc_score(y_test, y_skor)


# ==================================================
# 6. PERFORMANS TABLOSU
# ==================================================

metrikler = pd.DataFrame({
    "Metrik": [
        "Accuracy (Dogruluk)",
        "Precision (Kesinlik)",
        "Recall (Duyarlilik)",
        "F1-score",
        "ROC-AUC",
        "Egitim ilan sayisi",
        "Test ilan sayisi"
    ],
    "Deger": [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(roc_auc, 4),
        len(y_train),
        len(y_test)
    ]
})

metrikler.to_csv(
    SONUC_KLASORU / "performans_metrikleri.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ==================================================
# 7. HATA MATRISI TABLOSU
# ==================================================

hata_matrisi = pd.DataFrame(
    cm,
    index=["Gercek ilan (0)", "Sahte ilan (1)"],
    columns=[
        "Gercek tahmini (0)",
        "Sahte tahmini (1)"
    ]
)

hata_matrisi.index.name = "Gercek sinif"

hata_matrisi.to_csv(
    SONUC_KLASORU / "hata_matrisi_tablo.csv",
    sep=";",
    encoding="utf-8-sig"
)


# ==================================================
# 8. SINIFLANDIRMA RAPORU
# ==================================================

rapor = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Gercek ilan", "Sahte ilan"],
    output_dict=True,
    zero_division=0
)

rapor_df = pd.DataFrame(rapor).transpose()

rapor_df.to_csv(
    SONUC_KLASORU / "siniflandirma_raporu.csv",
    sep=";",
    encoding="utf-8-sig"
)


# ==================================================
# 9. TEST TAHMINLERI
# ==================================================

tahminler = pd.DataFrame({
    "Gercek_etiket": y_test,
    "Model_tahmini": y_pred,
    "Sahte_ilan_olasiligi": y_skor,
    "Sonuc": [
        "Dogru" if gercek == tahmin else "Yanlis"
        for gercek, tahmin in zip(y_test, y_pred)
    ]
})

tahminler.to_csv(
    SONUC_KLASORU / "tahminler.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ==================================================
# 10. EXCEL DOSYASI
# ==================================================

with pd.ExcelWriter(
    SONUC_KLASORU / "logistic_regression_tablolari.xlsx"
) as writer:

    metrikler.to_excel(
        writer,
        sheet_name="Performans Metrikleri",
        index=False
    )

    hata_matrisi.to_excel(
        writer,
        sheet_name="Hata Matrisi"
    )

    rapor_df.to_excel(
        writer,
        sheet_name="Siniflandirma Raporu"
    )

    tahminler.to_excel(
        writer,
        sheet_name="Test Tahminleri",
        index=False
    )


# ==================================================
# 11. GORSEL 1: CONFUSION MATRIX
# ==================================================

fig, ax = plt.subplots(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Gercek ilan", "Sahte ilan"]
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Logistic Regression - Confusion Matrix")
ax.set_xlabel("Tahmin edilen sinif")
ax.set_ylabel("Gercek sinif")

fig.tight_layout()

fig.savefig(
    SONUC_KLASORU / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ==================================================
# 12. GORSEL 2: PERFORMANS METRIKLERI
# ==================================================

isimler = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

degerler = [
    accuracy,
    precision,
    recall,
    f1
]

fig, ax = plt.subplots(figsize=(8, 5))

cubuklar = ax.bar(isimler, degerler)

ax.set_title("Logistic Regression - Performans Metrikleri")
ax.set_ylabel("Skor")
ax.set_ylim(0, 1.10)
ax.grid(axis="y", alpha=0.3)

for cubuk, deger in zip(cubuklar, degerler):
    ax.text(
        cubuk.get_x() + cubuk.get_width() / 2,
        deger + 0.02,
        f"{deger:.3f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()

fig.savefig(
    SONUC_KLASORU / "performans_grafigi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ==================================================
# 13. GORSEL 3: ROC EGRISI
# ==================================================

fpr, tpr, esikler = roc_curve(y_test, y_skor)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Rastgele tahmin"
)

ax.set_title("Logistic Regression - ROC Egrisi")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    SONUC_KLASORU / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ==================================================
# 14. MODELI KAYDET
# ==================================================

joblib.dump(
    model,
    SONUC_KLASORU / "logistic_regression_model.joblib"
)


# ==================================================
# 15. SONUC OZETINI DOSYAYA KAYDET
# ==================================================

ozet = f"""
LOGISTIC REGRESSION MODEL SONUCLARI

Egitim ilan sayisi: {len(y_train)}
Test ilan sayisi: {len(y_test)}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Hata matrisi:
{cm}

Sonuclar tablo ve grafik dosyalarina kaydedildi.
"""

(
    SONUC_KLASORU / "logistic_regression_sonuclari.txt"
).write_text(
    ozet,
    encoding="utf-8"
)
