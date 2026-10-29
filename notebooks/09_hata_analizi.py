# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================================================
# 1. DOSYA YOLLARI
# ==================================================

PROJE_KLASORU = Path(__file__).resolve().parent.parent
VERI_KLASORU = PROJE_KLASORU / "data" / "processed"
SONUC_KLASORU = PROJE_KLASORU / "reports" / "model_results"

SONUC_KLASORU.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. VERILERI YUKLE VE BIRLESTIR
# ==================================================

tahmin = pd.read_csv(
    SONUC_KLASORU / "nb_tahminler.csv", sep=";", encoding="utf-8-sig"
)
bolunme = pd.read_csv(VERI_KLASORU / "veri_bolunmesi.csv")
ilanlar = pd.read_csv(VERI_KLASORU / "balanced_cleaned.csv")

print("balanced_cleaned.csv sutunlari:", list(ilanlar.columns))


test_idler = bolunme.loc[bolunme["bolum"] == "test", "veri_satiri"].to_numpy()

assert len(test_idler) == len(tahmin), (
    "Test ilan sayisi ile tahmin sayisi uyusmuyor."
)
assert np.isin(test_idler, ilanlar.index).all(), (
    "veri_satiri degerleri balanced_cleaned.csv satirlariyla eslesmiyor."
)

test_ilanlar = ilanlar.loc[test_idler].reset_index(drop=True)
veri = pd.concat([tahmin.reset_index(drop=True), test_ilanlar], axis=1)
veri.insert(0, "veri_satiri", test_idler)

# Gercek etiket kontrolu: tahmin dosyasi ile ilan verisi ayni mi?
assert (veri["Gercek_etiket"].to_numpy() ==
        veri["fraudulent"].astype(int).to_numpy()).all(), (
    "Etiketler uyusmuyor, siralama bozulmus olabilir."
)


# ==================================================
# 3. HATA GRUPLARI
# ==================================================

veri["NB_yanlis"] = veri["NB_tahmini"] != veri["Gercek_etiket"]
veri["LR_yanlis"] = veri["LR_tahmini"] != veri["Gercek_etiket"]


def grup_bul(satir):
    if satir["NB_yanlis"] and satir["LR_yanlis"]:
        return "Ikisi de yanlis"
    if satir["NB_yanlis"]:
        return "Sadece NB yanlis"
    if satir["LR_yanlis"]:
        return "Sadece LR yanlis"
    return "Ikisi de dogru"


def hata_turu(gercek, tahmin_):
    if gercek == tahmin_:
        return "Dogru"
    return "Yanlis alarm (gercek -> sahte)" if gercek == 0 else "Kacirilan sahte (sahte -> gercek)"


veri["Grup"] = veri.apply(grup_bul, axis=1)
veri["NB_hata_turu"] = [hata_turu(g, t) for g, t in zip(veri["Gercek_etiket"], veri["NB_tahmini"])]
veri["LR_hata_turu"] = [hata_turu(g, t) for g, t in zip(veri["Gercek_etiket"], veri["LR_tahmini"])]

grup_sirasi = ["Ikisi de dogru", "Sadece NB yanlis", "Sadece LR yanlis", "Ikisi de yanlis"]
grup_sayilari = (
    veri["Grup"].value_counts().reindex(grup_sirasi).fillna(0).astype(int)
)

print("\nHATA GRUPLARI (test kumesi: {} ilan)".format(len(veri)))
print(grup_sayilari.to_string())

hata_turu_tablo = pd.DataFrame({
    "Naive Bayes": veri["NB_hata_turu"].value_counts(),
    "Logistic Regression": veri["LR_hata_turu"].value_counts(),
}).fillna(0).astype(int)
print("\nHATA TURLERI\n", hata_turu_tablo.to_string())


# ==================================================
# 4. METIN UZUNLUGU ANALIZI
# ==================================================

veri["Kelime_sayisi"] = (
    veri["clean_text"].fillna("").astype(str).str.split().str.len()
)

uzunluk_tablo = (
    veri.groupby("Grup")["Kelime_sayisi"]
    .agg(["count", "mean", "median", "min", "max"])
    .reindex(grup_sirasi)
    .round(1)
)
print("\nGRUPLARA GORE KELIME SAYISI\n", uzunluk_tablo.to_string())


dilimler = pd.qcut(
    veri["Kelime_sayisi"], q=4, duplicates="drop",
)
veri["Uzunluk_dilimi"] = dilimler.astype(str)

dilim_tablo = veri.groupby("Uzunluk_dilimi", sort=False).agg(
    Ilan_sayisi=("Gercek_etiket", "size"),
    NB_hata_orani=("NB_yanlis", "mean"),
    LR_hata_orani=("LR_yanlis", "mean"),
).round(3)


dilim_siralama = {str(d): i for i, d in enumerate(sorted(dilimler.dropna().unique()))}
dilim_tablo = dilim_tablo.loc[sorted(dilim_tablo.index, key=lambda d: dilim_siralama.get(d, 99))]
print("\nMETIN UZUNLUGU DILIMINE GORE HATA ORANI\n", dilim_tablo.to_string())


# ==================================================
# 5. EKSIK ALAN ANALIZI
# ==================================================

aday_alanlar = [
    "title", "company_profile", "description", "requirements", "benefits",
    "salary_range", "department", "required_education",
]
mevcut_alanlar = [a for a in aday_alanlar if a in veri.columns]

eksik_tablo = None
if mevcut_alanlar:
    satirlar = {}
    for alan in mevcut_alanlar:
        eksik = veri[alan].isna() | (veri[alan].astype(str).str.strip() == "")
        satirlar[alan] = {
            g: round(eksik[veri["Grup"] == g].mean() * 100, 1)
            for g in grup_sirasi if (veri["Grup"] == g).any()
        }
    eksik_tablo = pd.DataFrame(satirlar).T
    print("\nGRUPLARA GORE EKSIK ALAN ORANI (%)\n", eksik_tablo.to_string())
else:
    print("\nEksik alan analizi atlandi: ham metin sutunlari bulunamadi.")


# ==================================================
# 6. ORNEK YANLIS SINIFLANDIRILAN ILANLAR
# ==================================================


def kisa_metin(satir, uzunluk=250):
    metin = str(satir.get("title", "")) if "title" in satir.index else ""
    govde = str(satir.get("clean_text", ""))
    return (metin + " | " + govde)[:uzunluk].replace("\n", " ") if metin and metin != "nan" else govde[:uzunluk]


hatalilar = veri[veri["NB_yanlis"] | veri["LR_yanlis"]].copy()
hatalilar["Ozet_metin"] = hatalilar.apply(kisa_metin, axis=1)

kaydet_sutunlar = [
    "veri_satiri", "Gercek_etiket", "Grup",
    "NB_tahmini", "NB_sahte_ilan_olasiligi", "NB_hata_turu",
    "LR_tahmini", "LR_sahte_ilan_olasiligi", "LR_hata_turu",
    "Kelime_sayisi", "Ozet_metin",
]
hatalilar[kaydet_sutunlar].sort_values(["Grup", "veri_satiri"]).to_csv(
    SONUC_KLASORU / "hata_analizi_yanlis_ilanlar.csv",
    index=False, sep=";", encoding="utf-8-sig",
)

print("\n--- IKISININ DE YANLIS BILDIGI ILK ORNEKLER ---")
for _, s in hatalilar[hatalilar["Grup"] == "Ikisi de yanlis"].head(8).iterrows():
    etiket = "SAHTE" if s["Gercek_etiket"] == 1 else "GERCEK"
    print(f"[{etiket}] satir={s['veri_satiri']}  kelime={s['Kelime_sayisi']}  "
          f"NB_p={s['NB_sahte_ilan_olasiligi']:.2f}  LR_p={s['LR_sahte_ilan_olasiligi']:.2f}")
    print("   ", s["Ozet_metin"][:200])


# ==================================================
# 7. TABLOLARI KAYDET
# ==================================================

grup_tablo = pd.DataFrame({
    "Ilan sayisi": grup_sayilari,
    "Oran (%)": (grup_sayilari / len(veri) * 100).round(1),
})
grup_tablo.to_csv(
    SONUC_KLASORU / "hata_analizi_gruplar.csv",
    sep=";", encoding="utf-8-sig",
)
hata_turu_tablo.to_csv(
    SONUC_KLASORU / "hata_analizi_hata_turleri.csv",
    sep=";", encoding="utf-8-sig",
)
uzunluk_tablo.to_csv(
    SONUC_KLASORU / "hata_analizi_kelime_sayisi.csv",
    sep=";", encoding="utf-8-sig",
)
dilim_tablo.to_csv(
    SONUC_KLASORU / "hata_analizi_uzunluk_dilimleri.csv",
    sep=";", encoding="utf-8-sig",
)
if eksik_tablo is not None:
    eksik_tablo.to_csv(
        SONUC_KLASORU / "hata_analizi_eksik_alanlar.csv",
        sep=";", encoding="utf-8-sig",
    )

with pd.ExcelWriter(SONUC_KLASORU / "hata_analizi_tablolari.xlsx") as writer:
    grup_tablo.to_excel(writer, sheet_name="Hata Gruplari")
    hata_turu_tablo.to_excel(writer, sheet_name="Hata Turleri")
    uzunluk_tablo.to_excel(writer, sheet_name="Kelime Sayisi")
    dilim_tablo.to_excel(writer, sheet_name="Uzunluk Dilimleri")
    if eksik_tablo is not None:
        eksik_tablo.to_excel(writer, sheet_name="Eksik Alanlar")
    hatalilar[kaydet_sutunlar].to_excel(writer, sheet_name="Yanlis Ilanlar", index=False)


# ==================================================
# 8. GORSEL 1: HATA GRUPLARI
# ==================================================

fig, ax = plt.subplots(figsize=(8, 5))
cubuklar = ax.bar(grup_sirasi, grup_sayilari.values)
for c in cubuklar:
    ax.text(c.get_x() + c.get_width() / 2, c.get_height() + 2,
            str(int(c.get_height())), ha="center", va="bottom")
ax.set_title("Test Ilanlarinin Model Dogrulugu Gruplari")
ax.set_ylabel("Ilan sayisi")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "hata_analizi_gruplar.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 9. GORSEL 2: METIN UZUNLUGUNA GORE HATA ORANI
# ==================================================

x = np.arange(len(dilim_tablo))
genislik = 0.38

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - genislik / 2, dilim_tablo["LR_hata_orani"] * 100,
            genislik, label="Logistic Regression")
b2 = ax.bar(x + genislik / 2, dilim_tablo["NB_hata_orani"] * 100,
            genislik, label="Naive Bayes")
for grup in (b1, b2):
    for c in grup:
        ax.text(c.get_x() + c.get_width() / 2, c.get_height() + 0.3,
                f"{c.get_height():.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"Dilim {i + 1}\n{d}" for i, d in enumerate(dilim_tablo.index)],
                   fontsize=8)
ax.set_ylabel("Hata orani (%)")
ax.set_title("Metin Uzunlugu Dilimine Gore Hata Orani (kelime sayisi)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(SONUC_KLASORU / "hata_analizi_uzunluk_hata_orani.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================
# 10. OZET DOSYASI
# ==================================================

ozet = f"""
HATA ANALIZI OZETI (4. HAFTA)

Test ilan sayisi: {len(veri)}

Gruplar:
{grup_tablo.to_string()}

Hata turleri:
{hata_turu_tablo.to_string()}

Gruplara gore kelime sayisi:
{uzunluk_tablo.to_string()}

Metin uzunlugu dilimine gore hata orani:
{dilim_tablo.to_string()}
"""
if eksik_tablo is not None:
    ozet += f"\nGruplara gore eksik alan orani (%):\n{eksik_tablo.to_string()}\n"

(SONUC_KLASORU / "hata_analizi_ozeti.txt").write_text(ozet, encoding="utf-8")
print("\nTum ciktilar kaydedildi:", SONUC_KLASORU)
