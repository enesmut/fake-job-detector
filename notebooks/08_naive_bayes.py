# -*- coding: utf-8 -*-

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


# ==================================================
# 1. DOSYA YOLLARI
# ==================================================

PROJE_KLASORU = Path(__file__).resolve().parent.parent
VERI_KLASORU = PROJE_KLASORU / "data" / "processed"
SONUC_KLASORU = PROJE_KLASORU / "reports" / "model_results"

SONUC_KLASORU.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==================================================
# 2. VERILERI YUKLE
# ==================================================

X_train = load_npz(VERI_KLASORU / "X_train_tfidf.npz")
X_test = load_npz(VERI_KLASORU / "X_test_tfidf.npz")

bolunme = pd.read_csv(VERI_KLASORU / "veri_bolunmesi.csv")

y_train = bolunme.loc[
    bolunme["bolum"] == "egitim", "fraudulent"
].astype(int).to_numpy()

y_test = bolunme.loc[
    bolunme["bolum"] == "test", "fraudulent"
].astype(int).to_numpy()

assert X_train.shape[0] == len(y_train), "Egitim verisi ile etiket sayisi uyusmuyor."
assert X_test.shape[0] == len(y_test), "Test verisi ile etiket sayisi uyusmuyor."

print(f"Egitim matrisi: {X_train.shape}, Test matrisi: {X_test.shape}")


def metrik_hesapla(y_gercek, y_tahmin, y_skor):
    return {
        "Accuracy": accuracy_score(y_gercek, y_tahmin),
        "Precision": precision_score(y_gercek, y_tahmin, zero_division=0),
        "Recall": recall_score(y_gercek, y_tahmin, zero_division=0),
        "F1-score": f1_score(y_gercek, y_tahmin, zero_division=0),
        "ROC-AUC": roc_auc_score(y_gercek, y_skor),
    }


# ==================================================
# 3. ILK NAIVE BAYES MODELI (varsayilan alpha=1.0)
# ==================================================

t0 = time.time()
nb_ilk = MultinomialNB()
nb_ilk.fit(X_train, y_train)
nb_ilk_sure = time.time() - t0

nb_ilk_tahmin = nb_ilk.predict(X_test)
nb_ilk_skor = nb_ilk.predict_proba(X_test)[:, 1]
nb_ilk_sonuc = metrik_hesapla(y_test, nb_ilk_tahmin, nb_ilk_skor)


# ==================================================
# 4. HIPERPARAMETRE OPTIMIZASYONU (alpha, 5 katli CV, F1)
# ==================================================

arama = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid={"alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
arama.fit(X_train, y_train)

cv_tablo = (
    pd.DataFrame(arama.cv_results_)[
        ["param_alpha", "mean_test_score", "std_test_score"]
    ]
    .rename(columns={
        "param_alpha": "alpha",
        "mean_test_score": "CV_F1_Ortalamasi",
        "std_test_score": "CV_F1_Standart_Sapma",
    })
    .sort_values("CV_F1_Ortalamasi", ascending=False)
    .reset_index(drop=True)
)
cv_tablo = cv_tablo.round(4)

cv_tablo.to_csv(
    SONUC_KLASORU / "nb_alpha_denemeleri.csv",
    index=False, sep=";", encoding="utf-8-sig",
)

en_iyi_alpha = float(arama.best_params_["alpha"])

print("\nDenemelerin en iyi sonuclari:")
print(cv_tablo.head(5).to_string(index=False))
print("\nSecilen parametreler:", arama.best_params_)
print("Egitim verisi 5 katli CV F1:", round(arama.best_score_, 4))


# ==================================================
# 5. IYILESTIRILMIS NAIVE BAYES MODELI
# ==================================================

t0 = time.time()
model = MultinomialNB(alpha=en_iyi_alpha)
model.fit(X_train, y_train)
nb_sure = time.time() - t0

y_pred = model.predict(X_test)
y_skor = model.predict_proba(X_test)[:, 1]
nb_sonuc = metrik_hesapla(y_test, y_pred, y_skor)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Istege bagli: ComplementNB (karsilastirma icin)
cnb = ComplementNB(alpha=en_iyi_alpha).fit(X_train, y_train)
cnb_sonuc = metrik_hesapla(
    y_test, cnb.predict(X_test), cnb.predict_proba(X_test)[:, 1]
)


# ==================================================
# 6. LOGISTIC REGRESSION (3. hafta en iyi parametreler)
# ==================================================

t0 = time.time()
lr = LogisticRegression(
    C=10.0, penalty="l2", max_iter=1000, random_state=RANDOM_STATE
)
lr.fit(X_train, y_train)
lr_sure = time.time() - t0

lr_pred = lr.predict(X_test)
lr_skor = lr.predict_proba(X_test)[:, 1]
lr_sonuc = metrik_hesapla(y_test, lr_pred, lr_skor)
lr_cm = confusion_matrix(y_test, lr_pred, labels=[0, 1])


# ==================================================
# 7. NAIVE BAYES TABLOLARI
# ==================================================

metrikler = pd.DataFrame({
    "Metrik": [
        "Accuracy (Dogruluk)", "Precision (Kesinlik)", "Recall (Duyarlilik)",
        "F1-score", "ROC-AUC", "Egitim ilan sayisi", "Test ilan sayisi",
    ],
    "Ilk model (alpha=1.0)": [
        round(nb_ilk_sonuc["Accuracy"], 4), round(nb_ilk_sonuc["Precision"], 4),
        round(nb_ilk_sonuc["Recall"], 4), round(nb_ilk_sonuc["F1-score"], 4),
        round(nb_ilk_sonuc["ROC-AUC"], 4), len(y_train), len(y_test),
    ],
    f"Iyilestirilmis model (alpha={en_iyi_alpha})": [
        round(nb_sonuc["Accuracy"], 4), round(nb_sonuc["Precision"], 4),
        round(nb_sonuc["Recall"], 4), round(nb_sonuc["F1-score"], 4),
        round(nb_sonuc["ROC-AUC"], 4), len(y_train), len(y_test),
    ],
})
metrikler.to_csv(
    SONUC_KLASORU / "nb_performans_metrikleri.csv",
    index=False, sep=";", encoding="utf-8-sig",
)

hata_matrisi = pd.DataFrame(
    cm,
    index=["Gercek ilan (0)", "Sahte ilan (1)"],
    columns=["Gercek tahmini (0)", "Sahte tahmini (1)"],
)
hata_matrisi.index.name = "Gercek sinif"
hata_matrisi.to_csv(
    SONUC_KLASORU / "nb_hata_matrisi_tablo.csv",
    sep=";", encoding="utf-8-sig",
)

rapor_df = pd.DataFrame(
    classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Gercek ilan", "Sahte ilan"],
        output_dict=True, zero_division=0,
    )
).transpose()
rapor_df.to_csv(
    SONUC_KLASORU / "nb_siniflandirma_raporu.csv",
    sep=";", encoding="utf-8-sig",
)

tahminler = pd.DataFrame({
    "Gercek_etiket": y_test,
    "NB_tahmini": y_pred,
    "NB_sahte_ilan_olasiligi": y_skor,
    "NB_sonuc": ["Dogru" if g == t else "Yanlis" for g, t in zip(y_test, y_pred)],
    "LR_tahmini": lr_pred,
    "LR_sahte_ilan_olasiligi": lr_skor,
    "LR_sonuc": ["Dogru" if g == t else "Yanlis" for g, t in zip(y_test, lr_pred)],
})
tahminler.to_csv(
    SONUC_KLASORU / "nb_tahminler.csv",
    index=False, sep=";", encoding="utf-8-sig",
)


# ==================================================
# 8. KARSILASTIRMA TABLOSU (NB - LR)
# ==================================================

karsilastirma = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Logistic Regression": [lr_sonuc[m] for m in
                            ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]],
    "Naive Bayes": [nb_sonuc[m] for m in
                    ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]],
    "ComplementNB": [cnb_sonuc[m] for m in
                     ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]],
})
karsilastirma["Fark (NB - LR)"] = (
    karsilastirma["Naive Bayes"] - karsilastirma["Logistic Regression"]
)
karsilastirma = karsilastirma.round(4)

sure_tablo = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Egitim suresi (sn)": [round(lr_sure, 4), round(nb_sure, 4)],
})

karsilastirma.to_csv(
    SONUC_KLASORU / "nb_lr_karsilastirma.csv",
    index=False, sep=";", encoding="utf-8-sig",
)


# ==================================================
# 9. EXCEL DOSYASI
# ==================================================

with pd.ExcelWriter(SONUC_KLASORU / "naive_bayes_tablolari.xlsx") as writer:
    metrikler.to_excel(writer, sheet_name="Performans Metrikleri", index=False)
    cv_tablo.to_excel(writer, sheet_name="Alpha Denemeleri", index=False)
    hata_matrisi.to_excel(writer, sheet_name="Hata Matrisi")
    rapor_df.to_excel(writer, sheet_name="Siniflandirma Raporu")
    karsilastirma.to_excel(writer, sheet_name="NB-LR Karsilastirma", index=False)
    sure_tablo.to_excel(writer, sheet_name="Egitim Suresi", index=False)
    tahminler.to_excel(writer, sheet_name="Test Tahminleri", index=False)


# ==================================================
# 10. GORSEL 1: CONFUSION MATRIX (Naive Bayes)
# ==================================================

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Gercek ilan", "Sahte ilan"]
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Naive Bayes - Confusion Matrix")
ax.set_xlabel("Tahmin edilen sinif")
ax.set_ylabel("Gercek sinif")
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "nb_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 11. GORSEL 2: PERFORMANS METRIKLERI (Naive Bayes)
# ==================================================

isimler = ["Accuracy", "Precision", "Recall", "F1-score"]
degerler = [nb_sonuc[i] for i in isimler]

fig, ax = plt.subplots(figsize=(8, 5))
cubuklar = ax.bar(isimler, degerler)
ax.set_title("Naive Bayes - Performans Metrikleri")
ax.set_ylabel("Skor")
ax.set_ylim(0, 1.10)
ax.grid(axis="y", alpha=0.3)
for cubuk, deger in zip(cubuklar, degerler):
    ax.text(cubuk.get_x() + cubuk.get_width() / 2, deger + 0.02,
            f"{deger:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "nb_performans_grafigi.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 12. GORSEL 3: ROC EGRISI (Naive Bayes)
# ==================================================

nb_fpr, nb_tpr, _ = roc_curve(y_test, y_skor)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_sonuc['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Rastgele tahmin")
ax.set_title("Naive Bayes - ROC Egrisi")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "nb_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 13. GORSEL 4: NB - LR KARSILASTIRMA GRAFIGI
# ==================================================

metrik_adlari = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(metrik_adlari))
genislik = 0.38

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - genislik / 2, [lr_sonuc[m] for m in metrik_adlari],
            genislik, label="Logistic Regression")
b2 = ax.bar(x + genislik / 2, [nb_sonuc[m] for m in metrik_adlari],
            genislik, label="Naive Bayes")
for grup in (b1, b2):
    for cubuk in grup:
        ax.text(cubuk.get_x() + cubuk.get_width() / 2, cubuk.get_height() + 0.01,
                f"{cubuk.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metrik_adlari)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Skor")
ax.set_title("Logistic Regression ve Naive Bayes Karsilastirmasi")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "nb_lr_karsilastirma_grafigi.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 14. GORSEL 5: ORTAK ROC EGRISI
# ==================================================

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_skor)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_sonuc['ROC-AUC']:.3f})")
ax.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_sonuc['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Rastgele tahmin")
ax.set_title("ROC Egrisi Karsilastirmasi")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "nb_lr_ortak_roc.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 15. GORSEL 6: LR HATA MATRISI (karsilastirma icin)
# ==================================================

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=lr_cm, display_labels=["Gercek ilan", "Sahte ilan"]
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Logistic Regression - Confusion Matrix")
ax.set_xlabel("Tahmin edilen sinif")
ax.set_ylabel("Gercek sinif")
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "lr_confusion_matrix_karsilastirma.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 16. EN AYIRT EDICI KELIMELER
# ==================================================

try:
    vektorleyici = joblib.load(VERI_KLASORU / "tfidf_vektorleyici.joblib")
    kelimeler = np.array(vektorleyici.get_feature_names_out())

    # NB: log olasilik farki (pozitif = sahte ilana ozgu)
    nb_fark = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    # LR: katsayilar (pozitif = sahte ilana ozgu)
    lr_katsayi = lr.coef_[0]

    def ilk_n(degerler, n=20):
        sahte = kelimeler[np.argsort(degerler)[::-1][:n]]
        gercek = kelimeler[np.argsort(degerler)[:n]]
        return sahte, gercek

    nb_sahte, nb_gercek = ilk_n(nb_fark)
    lr_sahte, lr_gercek = ilk_n(lr_katsayi)

    kelime_tablo = pd.DataFrame({
        "NB_sahte_ilani_gosteren": nb_sahte,
        "NB_gercek_ilani_gosteren": nb_gercek,
        "LR_sahte_ilani_gosteren": lr_sahte,
        "LR_gercek_ilani_gosteren": lr_gercek,
    })
    kelime_tablo.to_csv(
        SONUC_KLASORU / "ayirt_edici_kelimeler.csv",
        index=False, sep=";", encoding="utf-8-sig",
    )

    print("\nNB - sahte ilana ozgu ilk 10 kelime:", list(nb_sahte[:10]))
    print("NB - gercek ilana ozgu ilk 10 kelime:", list(nb_gercek[:10]))
except Exception as hata:
    print("\nAyirt edici kelime analizi atlandi:", hata)


# ==================================================
# 17. MODELI KAYDET
# ==================================================

joblib.dump(model, SONUC_KLASORU / "naive_bayes_model.joblib")


# ==================================================
# 18. SONUC OZETINI KAYDET VE GOSTER
# ==================================================

hedef_f1 = nb_sonuc["F1-score"] >= 0.85
hedef_acc = nb_sonuc["Accuracy"] >= 0.85

ozet = f"""
NAIVE BAYES MODEL SONUCLARI (4. HAFTA)

Egitim ilan sayisi: {len(y_train)}
Test ilan sayisi: {len(y_test)}
Secilen alpha: {en_iyi_alpha}  (5 katli CV F1: {arama.best_score_:.4f})

ILK MODEL (alpha=1.0)
Accuracy: {nb_ilk_sonuc['Accuracy']:.4f}  Precision: {nb_ilk_sonuc['Precision']:.4f}
Recall: {nb_ilk_sonuc['Recall']:.4f}  F1-score: {nb_ilk_sonuc['F1-score']:.4f}  ROC-AUC: {nb_ilk_sonuc['ROC-AUC']:.4f}

IYILESTIRILMIS MODEL (alpha={en_iyi_alpha})
Accuracy: {nb_sonuc['Accuracy']:.4f}  Precision: {nb_sonuc['Precision']:.4f}
Recall: {nb_sonuc['Recall']:.4f}  F1-score: {nb_sonuc['F1-score']:.4f}  ROC-AUC: {nb_sonuc['ROC-AUC']:.4f}

Hata matrisi:
{cm}

LOGISTIC REGRESSION (C=10, l2) ile KARSILASTIRMA
{karsilastirma.to_string(index=False)}

Egitim suresi (sn): LR={lr_sure:.4f}  NB={nb_sure:.4f}
Proje hedefi (>= 0.85): F1 -> {hedef_f1}, Accuracy -> {hedef_acc}
"""

(SONUC_KLASORU / "naive_bayes_sonuclari.txt").write_text(ozet, encoding="utf-8")
print(ozet)
print("Tum ciktilar kaydedildi:", SONUC_KLASORU)
