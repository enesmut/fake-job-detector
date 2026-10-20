
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# 1. DOSYA YOLLARI
# ==================================================

PROJE_KLASORU = Path(__file__).resolve().parent.parent
VERI_KLASORU = PROJE_KLASORU / "data" / "processed"
SONUC_KLASORU = (
    PROJE_KLASORU / "reports" / "model_results"
)

SONUC_KLASORU.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. VERILERI YUKLE
# ==================================================

X_train = load_npz(
    VERI_KLASORU / "X_train_tfidf.npz"
)

X_test = load_npz(
    VERI_KLASORU / "X_test_tfidf.npz"
)

bolunme = pd.read_csv(
    VERI_KLASORU / "veri_bolunmesi.csv"
)

y_train = bolunme.loc[
    bolunme["bolum"] == "egitim",
    "fraudulent"
].astype(int).to_numpy()

y_test = bolunme.loc[
    bolunme["bolum"] == "test",
    "fraudulent"
].astype(int).to_numpy()


# ==================================================
# 3. ILK MODELIN SONUCLARINI AL
# ==================================================

ilk_sonuc_dosyasi = (
    SONUC_KLASORU / "performans_metrikleri.csv"
)

ilk_metrikler_df = pd.read_csv(
    ilk_sonuc_dosyasi,
    sep=";"
)

ilk_metrikler = dict(
    zip(
        ilk_metrikler_df["Metrik"],
        ilk_metrikler_df["Deger"]
    )
)

ilk_accuracy = float(
    ilk_metrikler["Accuracy (Dogruluk)"]
)

ilk_precision = float(
    ilk_metrikler["Precision (Kesinlik)"]
)

ilk_recall = float(
    ilk_metrikler["Recall (Duyarlilik)"]
)

ilk_f1 = float(
    ilk_metrikler["F1-score"]
)

ilk_roc_auc = float(
    ilk_metrikler["ROC-AUC"]
)


# ==================================================
# 4. PARAMETRE ARAMASI
# ==================================================

parametreler = {
    "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "penalty": ["l1", "l2"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LogisticRegression(
    solver="liblinear",
    max_iter=2000,
    random_state=42
)

arama = GridSearchCV(
    estimator=model,
    param_grid=parametreler,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

arama.fit(X_train, y_train)


# ==================================================
# 5. PARAMETRE SONUCLARINI TABLOYA DONUSTUR
# ==================================================

cv_sonuclari = pd.DataFrame(
    arama.cv_results_
)

parametre_tablosu = cv_sonuclari[
    [
        "param_C",
        "param_penalty",
        "mean_test_score",
        "std_test_score"
    ]
].copy()

parametre_tablosu.columns = [
    "C",
    "Duzenlilestirme",
    "CV_F1_Ortalamasi",
    "CV_F1_Standart_Sapma"
]

parametre_tablosu = parametre_tablosu.sort_values(
    "CV_F1_Ortalamasi",
    ascending=False
).reset_index(drop=True)


# ==================================================
# 6. EN IYI MODELI TEST VERISINDE DEGERLENDIR
# ==================================================

en_iyi_model = arama.best_estimator_

y_pred = en_iyi_model.predict(X_test)

y_skor = en_iyi_model.predict_proba(X_test)[:, 1]

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

roc_auc = roc_auc_score(y_test, y_skor)


# ==================================================
# 7. ILK MODEL VE YENI MODELI KARSILASTIR
# ==================================================

karsilastirma = pd.DataFrame({
    "Metrik": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Ilk_Model": [
        ilk_accuracy,
        ilk_precision,
        ilk_recall,
        ilk_f1,
        ilk_roc_auc
    ],
    "Iyilestirilmis_Model": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

karsilastirma["Fark"] = (
    karsilastirma["Iyilestirilmis_Model"]
    - karsilastirma["Ilk_Model"]
)

karsilastirma["Ilk_Model"] = (
    karsilastirma["Ilk_Model"].round(4)
)

karsilastirma["Iyilestirilmis_Model"] = (
    karsilastirma["Iyilestirilmis_Model"].round(4)
)

karsilastirma["Fark"] = (
    karsilastirma["Fark"].round(4)
)


# ==================================================
# 8. GELISME VAR MI?
# ==================================================

gelisme_var = (
    f1 > ilk_f1
    or (
        f1 == ilk_f1
        and accuracy > ilk_accuracy
    )
)


# ==================================================
# 9. SONUCLARI TERMINALDE GOSTER
# ==================================================

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION - PARAMETRE IYILESTIRME")
print("=" * 65)

print("\nDenemelerin en iyi 5 sonucu:")
print(
    parametre_tablosu.head(5).to_string(
        index=False,
        formatters={
            "CV_F1_Ortalamasi": "{:.4f}".format,
            "CV_F1_Standart_Sapma": "{:.4f}".format
        }
    )
)

print("\nSecilen parametreler:")
print(arama.best_params_)

print(
    f"Egitim verisi 5 katli CV F1: "
    f"{arama.best_score_:.4f}"
)

print("\nILK MODEL - IYILESTIRILMIS MODEL")
print("-" * 65)

print(
    karsilastirma.to_string(
        index=False,
        formatters={
            "Ilk_Model": "{:.4f}".format,
            "Iyilestirilmis_Model": "{:.4f}".format,
            "Fark": "{:+.4f}".format
        }
    )
)

print("\n" + "=" * 65)

if gelisme_var:
    print("SONUC: F1-score veya esitlik durumunda Accuracy gelisti.")
    print("Iyilestirme adayi bulundu.")
else:
    print("SONUC: F1-score ve Accuracy birlikte iyilesmedi.")
    print("Mevcut model simdilik daha iyi olabilir.")

print("=" * 65)
