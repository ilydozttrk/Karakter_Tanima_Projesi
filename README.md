# Karakter Tanıma Projesi

Bu proje, görüntüler üzerinden karakter (harf ve/veya rakam) tanıma işlemi yapan Python tabanlı bir uygulamadır. Proje, Optik Karakter Tanıma (OCR) yaklaşımına dayanır ve görüntü işleme ile makine öğrenmesi tekniklerini bir arada kullanır. Amaç, bir görüntüde bulunan karakterleri otomatik olarak tespit etmek ve sınıflandırmaktır.

---

## İçindekiler

- Proje Hakkında  
- Kullanılan Teknolojiler  
- Özellikler  
- Kurulum  
- Kullanım  
- Proje Yapısı  
- Çalışma Mantığı  
- Sonuçlar ve Değerlendirme  
- Geliştirme Fikirleri  
- Katkıda Bulunma  
- Lisans  

---

## Proje Hakkında

Karakter Tanıma Projesi, sayısal ve alfabetik karakterleri tanıyabilen basit bir OCR sistemidir. Projede, görüntü işleme teknikleri ile karakterler görüntüden ayrıştırılır ve daha sonra eğitilmiş bir model yardımıyla hangi karakter oldukları tahmin edilir.

Bu proje:
- Bilgisayarlı görü (Computer Vision)  
- Makine öğrenmesi  
- Görüntü işleme  
alanlarında temel uygulama örneği sunar.

---

## Kullanılan Teknolojiler

- Python  
- OpenCV  
- NumPy  
- TensorFlow / Keras (isteğe bağlı)  
- Tkinter (arayüz için)  

---

## Özellikler

- Görüntü üzerinden karakter tanıma  
- Eğitim ve test aşamalarının ayrı olması  
- Rakamlar için ayrı eğitim dosyası  
- Basit grafiksel arayüz  
- Modelin kaydedilip tekrar kullanılabilmesi  

---

## Kurulum

Projeyi çalıştırmak için sırasıyla şu komutları çalıştırın:

```bash
git clone https://github.com/ilydozttrk/Karakter-Tan-ma-Projesi.git
cd Karakter_Tanima_Projesi
pip install opencv-python numpy tensorflow matplotlib
python egitim.py
python arayuz.py

```

## Kullanım

1. `egitim.py` dosyası çalıştırılarak model eğitilir.  
2. Eğitim tamamlandıktan sonra `arayuz.py` dosyası çalıştırılır.  
3. Açılan arayüz üzerinden görüntü seçilerek karakter tanıma işlemi yapılır.

---

## Proje Yapısı

```text
/
├── arayuz.py
├── egitim.py
├── egitim_rakam.py
├── dataset/
└── README.md
