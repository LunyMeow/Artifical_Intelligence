🧠 Adaptive Neural Cortex AI


Görsel: Dinamik olarak yeniden yapılandırılan bir nöral ağın temsili.

🚀 Proje Özeti

Bu proje, kendi nöral yapısını dinamik olarak düzenleyebilen bir yapay zeka sistemini sunmaktadır. Corteks sınıfı sayesinde, yapay zeka öğrenme sürecinde kendi mimarisini optimize edebilir, bu da onu geleneksel sabit yapılı modellere göre daha esnek ve adaptif kılar.

🧩 Temel Özellikler

Dinamik Nöral Yapı:
Corteks sınıfı, yapay zekanın nöron ve katman sayısını öğrenme sürecine göre ayarlamasına olanak tanır.

Modüler Tasarım:
Kod yapısı, farklı veri setleri ve problemler için kolayca uyarlanabilir modüllerden oluşur.

Eğitim ve Test Verileri:
Projede, digits_dataset.csv ve parity_problem.csv gibi çeşitli veri setleri kullanılarak modelin eğitimi ve testi gerçekleştirilmiştir.


🗂️ Proje Yapısı

Artifical_Intelligence/
├── neuronv*.py               # En güncel nöron yapısı
├── digits_dataset.csv        # Rakam tanıma veri seti
├── parity_problem.csv        # Parite problemi veri seti
├── requirements.txt          # Gerekli Python kütüphaneleri
└── README.md                 # Proje açıklamaları

🧠 Corteks Sınıfı Hakkında

Corteks sınıfı, yapay zekanın kendi nöral yapısını öğrenme sürecine göre yeniden yapılandırmasını sağlar. Bu, modelin performansını artırmak için katman sayısını ve nöron yapılarını dinamik olarak ayarlamasına olanak tanır.

class Corteks:
    def __init__(self, initial_structure):
        self.structure = initial_structure

    def adapt_structure(self, performance_metrics):
        # Performansa göre yapıyı güncelle
        if performance_metrics['accuracy'] < 0.8:
            self.structure['layers'] += 1
        # Diğer yapılandırma kuralları...

🛠️ Kurulum ve Kullanım

1. Gereksinimleri Yükleyin:

pip install -r requirements.txt


2. Modeli Eğitin:

python neuronv*.py
trainFromFile(veri.csv;katman,yapısı;lr değeri)
*eğitimi bekleyin*
set_input 0 0 1 0.4 3 #giriş verileri
refresh() #sonucu alın


3. Sonuçları İnceleyin:

Eğitim ve test sonuçları, konsola yazdırılır ve ayrıca bir log dosyasına kaydedilir.



📈 Örnek Sonuçlar

📌 Notlar

Proje, Python 3.8 veya üzeri sürümlerle uyumludur.

Geliştirme sürecinde numpy, pandas ve scikit-learn kütüphaneleri kullanılmıştır.
