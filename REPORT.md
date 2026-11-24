# Proje Raporu: Food-101 ile Yemek Fotoğrafı Sınıflandırma: Sıfırdan CNN ve ResNet-50 Transfer Learning

Bu tablo `food101_cnn.ipynb` tarafından `results/results.json`'dan otomatik üretilir.

| Alan | Açıklama / Öğrenci yanıtı |
|---|---|
| Project Title | Food-101 ile Yemek Fotoğrafı Sınıflandırma: Sıfırdan CNN ve ResNet-50 Transfer Learning |
| Problem / Task | Görüntü sınıflandırma: bir yemek fotoğrafını 101 yemek sınıfından birine atama |
| Dataset | Food-101 (101 sınıf, 101 000 görüntü). Bölme: train 68175, validation 7575 (resmî train'den sınıf başına 75, tabakalı), test 25250 (resmî test, yalnız sonda bir kez) |
| Dataset URL | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ |
| Model Architecture | ResNet-50 (ImageNet ön-eğitimli, ince ayar) (23,7 M parametre); karşılaştırma: Sıfırdan CNN (4 blok) |
| Input Size | 3 x 224 x 224 (RGB, ImageNet ortalama/std ile normalize) |
| Hyperparameters | epoch 6, batch 64, lr (head) 0,001, lr (gövde) 0,0001, weight decay 0,0001, warmup 0,5 epoch + kosinüs azalma, label smoothing 0,1, dropout 0,2, seed 42, karma hassasiyet (AMP fp16) |
| Optimizer | AdamW (gövde ve sınıflandırıcı başı için ayrı öğrenme oranı) |
| Activation / Output Function | Gizli katmanlar ReLU; çıkış 101 nöronlu Linear + Softmax |
| Loss Function | Cross-entropy (label smoothing 0,1) |
| Evaluation Metrics | Accuracy, top-5 accuracy, macro precision / recall / F1, karışıklık matrisi |
| Training Environment | Modal (L4, 8 vCPU), NVIDIA L4, PyTorch 2.15.0+cu132, eğitim süresi 14,6 dk |
| Best Performance | Test accuracy %86,99, top-5 %97,37, macro F1 %86,93 |
| Inference Time | Görüntü başına 7,7 ms (GPU, batch 1), 118 ms (CPU, batch 1) |
| Deployment Technology | Gradio web uygulaması (`app.py`): fotoğraf yükleme/kamera, ilk 5 tahmin, Grad-CAM |
| Repository | https://github.com/uuu4/bil485-cv-food101 |

## Model karşılaştırması (aynı train/validation/test bölmesi)

| Model | Parametre | En iyi epoch | Val acc | Test acc | Test top-5 | Test macro F1 | Eğitim süresi |
|---|---|---|---|---|---|---|---|
| Sıfırdan CNN (4 blok) | 1,2 M | 6 | %48,08 | %52,25 | %80,30 | %50,91 | 4,2 dk |
| ResNet-50 (ImageNet ön-eğitimli, ince ayar) | 23,7 M | 5 | %82,60 | %86,99 | %97,37 | %86,93 | 14,6 dk |

## Gerçek dünya fotoğrafları (Food-101 dışından)

| Dosya | Gerçek | Food-101'de var mı | Tahmin (olasılık) | Doğru mu | Gerçek sınıfın sırası |
|---|---|---|---|---|---|
| baklava.jpg | Baklava | evet | Baklava (%93,1) | evet | 1 |
| fried_calamari.jpg | Fried calamari | evet | Onion rings (%76,7) | hayır | 2 |
| hamburger.jpg | Hamburger | evet | Hamburger (%56,4) | evet | 1 |
| lahmacun.jpg | Lahmacun | hayır | Pizza (%63,2) | sınıf dışı | - |
| pizza.jpg | Pizza | evet | Pizza (%49,3) | evet | 1 |

## Değerlendirme

**Genel sonuç.** ResNet-50 ince ayarı test kümesinde %87,0 doğruluk verdi (top-5 %97,4, macro F1 %86,9); aynı veride sıfırdan eğitilen
küçük CNN %52,3'te kaldı. İki model aynı bölme, kayıp fonksiyonu ve öğrenme oranı takvimiyle eğitildiği için aradaki 35 puanlık fark
büyük ölçüde ImageNet ön-eğitiminden geliyor: ResNet-50 daha ilk epoch'ta %74 validation doğruluğuna ulaştı, sıfırdan CNN altı epoch
sonunda %48'deydi. §2'deki beklentiler (sıfırdan CNN %40–55, ResNet-50 %82–87) tuttu.

**Öğrenme eğrileri.** ResNet-50'de validation doğruluğu 4. epoch'tan sonra %82,5 civarında düzleşti, train doğruluğu ise %90'a
çıkmaya devam etti. Hafif aşırı öğrenme başlıyor; en iyi epoch bu yüzden 5. Train kaybının validation kaybından yüksek görünmesi hata
değil: train kaybı label smoothing'li ve çoğaltılmış (kırpılmış, çevrilmiş) görüntülerle ölçülüyor. Sıfırdan CNN'de iki eğri de hâlâ
yükseliyordu; daha uzun eğitimle artardı, ama ön-eğitimli modele yaklaşması beklenmez.

**Validation ile test arasındaki fark.** Test doğruluğu (%87,0) validation'dan (%82,6) yüksek. Validation, Food-101'in temizlenmemiş
train kümesinden ayrıldığı için yanlış etiketli örnekler içeriyor; test kümesi elle kontrol edilmiş. Validation'daki hataların bir
kısmı modelin değil etiketin hatası.

**Karışan sınıflar.** En çok karışan çiftler görünüşü neredeyse aynı yemekler: biftekler fileto mignon sanıldı (bifteklerin %16'sı),
çikolatalı kek ile çikolatalı mus iki yönde karıştı, ton balığı tartarı dana tartarı, prime rib biftek sanıldı. En düşük F1 de bu
gruptan: steak 0,59, pork chop 0,64, chocolate mousse 0,71. En kolay sınıflar kendine özgü şekli ya da rengi olanlar: edamame 0,99,
macarons 0,98. Bu hataların çoğunda ayırt edici bilgi (et türü, kesim, iç doku) fotoğrafta zaten zor görünüyor.

**Gerçek dünya fotoğrafları (5 adet, Wikimedia Commons, Food-101 dışından).** Food-101'de olan 4 yemeğin 3'ü doğru bilindi, dördü de
ilk 2 tahmin içinde.
- *Baklava* en emin tahmin (%93). Grad-CAM katmanlı hamur ve üçgen dilimlerin üzerinde.
- *Hamburger* doğru (%56), ama Grad-CAM yalnız köfte ve ekmek kenarında; patatesler karara neredeyse girmiyor. İkinci tahmin pulled
  pork sandwich (%23): model "ekmek arası et" görüyor.
- *Pizza* doğru ama yalnız %49 olasılıkla. Kare dilimler ve iri fesleğen yaprakları Food-101'deki tipik yuvarlak pizzaya benzemiyor.
- *Kalamar tava* yanlış: onion rings %77, doğru sınıf %9 ile ikinci. Grad-CAM halkaların üzerinde; şekil olarak ikisi aynı, ayıran
  ipucu (kesitteki doku) bu çözünürlükte kayboluyor.
- *Lahmacun* Food-101'de yok; model en yakın sınıfı söyledi: pizza (%63). Asıl önemli nokta, bu olasılığın uygulamadaki %40 uyarı
  eşiğinin üstünde olması. Model sınıf dışı bir yemeğe de yüksek güvenle cevap verebiliyor; softmax olasılığı "bu yemek tanıdığım
  sınıflardan değil" demeyi güvenilir biçimde bilmiyor.

**Çıkarım süresi.** ResNet-50 görüntü başına GPU'da 7,7 ms, CPU'da 118 ms sürüyor (batch 1, ön işleme hariç). Uygulama GPU'suz bir
sunucuda da beklemeden cevap verebilir.

**Sınırlılıklar.**
1. Tek tohumla tek koşu yapıldı; sonuçların koşudan koşuya değişimi ölçülmedi.
2. Hiperparametre araması yapılmadı; değerler ince ayar için yaygın başlangıç değerleri.
3. Gerçek dünya testi 5 fotoğraf, genelleme iddiası için az. Ödev belgesi telefonla çekilmiş fotoğraf istiyor; bu sürümde fotoğraflar
   açık lisanslı internet fotoğrafları (kaynaklar `my_photos/SOURCES.md`).
4. Sınıf dışı girdiyi ayırt eden bir mekanizma yok (ör. "bilinmeyen" sınıfı ya da güven kalibrasyonu); lahmacun örneği bunu gösteriyor.
5. Uygulama fp16 kaydedilmiş ağırlıkları yüklüyor; olasılıklar notebook'taki fp32 modelden en fazla 0,4 puan farklı (ör. pizza %49,7
   ve %49,3), tahmin edilen sınıflar aynı.

**Eğitim ortamı.** Notebook Modal'da (NVIDIA L4, 8 vCPU) papermill ile baştan sona çalıştırıldı (`tools/modal_run.py`); toplam 22 dakika,
ResNet-50 eğitimi 14,6 dakika. Aynı notebook Colab'de değişiklik yapmadan çalışır.
