# 1. Hafta -- Veri Setinin İncelenmesi ve Ön Analizler

## 1. Haftanın amacı

Bu haftanın amacı, sahte iş ilanlarının makine öğrenmesi ile tespit
edilmesi projesinde kullanılacak veri setini tanımak, veri kalitesini
incelemek ve metin verileri üzerinde ön analizler gerçekleştirmektir. Bu
aşamada herhangi bir makine öğrenmesi modeli eğitilmemiştir.

## 2. Veri setinin genel özellikleri

İncelenen `balanced.csv` dosyası 1.732 satır ve 18 sütundan
oluşmaktadır. Hedef değişken `fraudulent` sütunudur.

  Sınıf        Açıklama        İlan sayısı
  ------------ ------------- -------------
  0            Gerçek ilan             866
  1            Sahte ilan              866
  **Toplam**                     **1.732**

Sınıflar eşit sayıda temsil edilmektedir. Tamamen tekrarlanan satır
sayısı 0 olarak bulunmuştur.

## 3. Eksik verilerin incelenmesi

  Sütun                   Eksik değer sayısı   Eksik oranı
  --------------------- -------------------- -------------
  salary_range                         1.382        %79,79
  department                           1.102        %63,63
  required_education                     837        %48,33
  required_experience                    771        %44,52
  company_profile                        718        %41,45
  benefits                               702        %40,53
  function                               651        %37,59
  industry                               507        %29,27
  employment_type                        401        %23,15
  requirements                           276        %15,94
  location                                40         %2,31
  description                              1         %0,06
  job_id                                   0         %0,00
  telecommuting                            0         %0,00
  has_questions                            0         %0,00
  has_company_logo                         0         %0,00
  title                                    0         %0,00
  fraudulent                               0         %0,00

Eksik değerlerin özellikle `salary_range` ve `department` sütunlarında
yoğunlaştığı görülmüştür. Metin alanları ayrıca boş veya yalnızca boşluk
içeren kayıtlar açısından kontrol edilmiştir: `company_profile` için
718, `description` için 2, `requirements` için 276 ve `benefits` için
702 boş kayıt belirlenmiştir. `title` alanında boş kayıt bulunmamıştır.

`description` alanında eksik değer analizinde 1, metin kontrolünde ise 2
boş kayıt görülmüştür. `isnull()` eksik değerleri sayarken boş metinleri
aynı şekilde saymadığı için sonuçlar farklılaşabilir. Bu aşamada eksik
kayıtlar silinmemiştir; ilerleyen aşamada kullanılacak özelliklere göre
nasıl ele alınacaklarına karar verilecektir.

## 4. Açıklama uzunluklarının analizi

  --------------------------------------------------------------------------
  İlan      Kayıt sayısı     Ortalama       Medyan      En kısa      En uzun
  sınıfı                     karakter     karakter              
  --------- ------------ ------------ ------------ ------------ ------------
  Gerçek             866     1.208,51        1.048           60        5.303

  Sahte              866     1.154,83        844,5            0        8.578
  --------------------------------------------------------------------------

Gerçek ve sahte ilanların açıklama uzunlukları histogram ile
karşılaştırılmıştır. İki sınıfın dağılımları önemli ölçüde
örtüşmektedir. Sahte ilanlarda ortalama ve medyan uzunluk daha düşük
olsa da bu sınıfta çok uzun açıklamalar da vardır. Bu nedenle açıklama
uzunluğunun tek başına güvenilir bir sınıflandırma ölçütü olduğu
sonucuna varılamaz.

## 5. Kelime frekansı analizi

Gerçek ve sahte ilanlarda en sık geçen 20 kelime ayrı ayrı incelenmiş ve
grafiklerle görselleştirilmiştir. Her iki sınıfta da `and`, `to`, `the`,
`of` ve `in` gibi İngilizce kelimeler sık görülmüştür. Bu kelimeler
genel dil kullanımında da yaygın olduğundan ham kelime frekansları tek
başına ilanların gerçek veya sahte olduğunu belirlemeye yeterli
değildir. Analiz, metinleri tanımaya yönelik bir ön inceleme olarak
değerlendirilmiştir.

## 6. Dil dağılımı analizi

  Tespit sonucu                  Kayıt sayısı
  ---------------------------- --------------
  İngilizce (en)                        1.717
  Çok kısa metin (too_short)                7
  Yunanca (el)                              4
  Almanca (de)                              1
  İspanyolca (es)                           1
  Katalanca (ca)                            1
  Boş metin (empty)                         1
  **Toplam**                        **1.732**

İlanların yaklaşık %99,13'ü İngilizce olarak tespit edilmiştir. Bu
bulgu, ilk sürümde İngilizce metinlere odaklanmayı değerlendirmek için
dayanak oluşturmaktadır. Otomatik dil tespiti kesin değildir; özellikle
kısa metinlerde yanlış sınıflandırma ihtimali bulunduğundan diğer
etiketler kesin dil doğrulaması olarak yorumlanmamıştır.

## 7. Metin ön işleme denemesi

Örnek metinlerde küçük harfe dönüştürme, HTML etiketlerini temizleme,
noktalama işaretlerini boşluklarla değiştirme, alt çizgileri temizleme,
fazla boşlukları azaltma ve İngilizce stop word olarak belirlenen yaygın
kelimeleri kaldırma işlemleri denenmiştir. Eksik metin alanları boş
metin olarak ele alınmıştır.

Unicode karakterlerini koruyacak düzenleme ile Almanca örnekteki `ü` ve
`ß` gibi karakterlerin yanlışlıkla parçalanması önlenmiştir. Ancak
kullanılan stop word listesi İngilizce olduğu için diğer diller için
uygun bir yöntem olarak değerlendirilmemelidir. Bu çalışma ön işleme
yaklaşımının ilk denemesidir; bütün veri setinde nihai ön işleme süreci
tamamlanmamıştır.

## 8. Bu haftanın çıktıları

-   `notebooks/01_veri_inceleme.py`
-   `notebooks/02_metin_analizi.py`
-   `notebooks/03_kelime_analizi.py`
-   `notebooks/04_on_isleme_deneme.py`
-   `notebooks/05_dil_analizi.py`
-   `reports/aciklama_uzunluklari.png`
-   `reports/kelime_frekans_gercek.png`
-   `reports/kelime_frekans_sahte.png`
-   `reports/en_sik_kelimeler_gerçek.csv`
-   `reports/en_sik_kelimeler_sahte.csv`

## 9. Genel değerlendirme ve sonraki aşama

İlk hafta sonunda veri setinin boyutu, sınıf dağılımı, eksik değerleri,
açıklama uzunlukları, sık kullanılan kelimeleri ve dil dağılımı
incelenmiştir. Örnek metinler üzerinde temel metin temizleme adımları da
denenmiştir.

Bu aşamada makine öğrenmesi modelleri eğitilmemiş, model başarı
metrikleri hesaplanmamış ve kullanıcı arayüzü geliştirilmemiştir.
Sonraki aşamada kullanılacak metin alanlarının belirlenmesi, veri
hazırlama yaklaşımının netleştirilmesi ve eğitim-test ayrımı ile TF-IDF
tabanlı özellik çıkarımı için hazırlık yapılması planlanmaktadır.
