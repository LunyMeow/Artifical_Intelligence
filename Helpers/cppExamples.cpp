#include <iostream>
#include <fstream> // Dosya işlemleri için gerekli kütüphane
#include <string>
#include <cmath> // Matematik işlemleri için gerekli

#include <vector> // genişleyen liste için

#include <ctime> // zaman için

using namespace std;

#ifdef _WIN32
#include <windows.h> // Sleep(ms)
#else
#include <unistd.h> // usleep(microseconds)
#endif

// Platforma bağımlı bekleme fonksiyonu
void bekle(int saniye)
{
#ifdef _WIN32
    Sleep(saniye * 1000); // Windows: milisaniye
#else
    usleep(saniye * 1000000); // Linux/MacOS: mikro saniye
#endif
}

// Şimdiki zamanı ekrana yazdırma fonksiyonu
void simdikiZaman()
{
    std::time_t t = std::time(nullptr); // şimdiki zaman
    std::tm *tmPtr = std::localtime(&t);

    std::cout << "Tarih ve Saat: "
              << tmPtr->tm_mday << "/"
              << tmPtr->tm_mon + 1 << "/" // ay 0-11 arası
              << tmPtr->tm_year + 1900 << " "
              << tmPtr->tm_hour << ":"
              << tmPtr->tm_min << ":"
              << tmPtr->tm_sec << std::endl;
}

int toplam(int a, int b)
{
    return a + b;
}

void biri_bana_seslendi()
{
    cout << "Biri bana mi seslendi?" << endl;
}

void selamVer(string isim = "dünya", string selam = "Merhaba ")
{
    cout << selam << isim << endl;
}

void sepetF(string sepet[4])
{
    // cout <<"Sepetin boyutu :"<< sizeof(sepet) << endl; // Hata verir
    for (int i = 0; i < 4; i++) // fonksiyonlarda parametrelerin boyutu için sizeof kullanamzsın pointer değerini verir 4 veya 8 boyte yani
    {
        cout << "Sepetinde :" << sepet[i] << endl;
    }
}

class HesapMakinesi
{
private:
    double privateNum1;
    double privateNum2;

public:
    double num1;
    double num2;

    HesapMakinesi()
    {
        cout << "Hesap makinesi sinifi belirlendi..." << endl;
    }

    double topla(int a, int b)
    {
        return a + b;
    }

    double cikart(int a, int b);
};

double HesapMakinesi ::cikart(int a, int b) // Bu fonksiyon sınıfın içine sonradan eklendi
{
    return a - b;
}

class IleriHesaplama : public HesapMakinesi
{
public:
    double kokAl(int num1, int num2)
    {
        return pow(num1, 1.0 / num2); // 1.0 yazmamın sebebi 0 a bölme hatası vermemesi için
    }
    double usAl(int num1, int num2)
    {
        return pow(num1, num2);
    }
};
/*3️⃣ Küçük notlar

sqrt(x) → karekök almak için kısayol (aynı şey pow(x, 0.5) ile).

cbrt(x) → küpkök almak için kısayol (aynı şey pow(x, 1.0/3) ile). */

int main()
{
    /*

    cout << "Hello World!\n\n";
    int a;
    cout << "Ilk sayiyi giriniz:";
    cin >> a;
    int b;
    cout << "Ikinci sayiyi giriniz:";
    cin >> b;

    cout << "Girilen sayilarin toplami:" << toplam(a, b) << "\n\n";

    string isim;
    cout << "Adinizi giriniz:";
    // cin >> isim; //Cin kullanırsan yalnzıca kelimeyi alır boşluklardan sonrasını almaz
    cin.ignore(); // daha önce kullanılan cinlerde \n olduğu için bunu kullanmalıyız
    getline(cin, isim);

    cout << "Merhaba " << isim << "\n";
    string meyveler[2] = {"Armut", "Elma"};

    cout << "Listenin ramda kapladigi alan:" << sizeof(meyveler) << endl;
    cout << "Listenin eleman sayi:" << sizeof(meyveler) / sizeof(meyveler[0]) << endl;
    for (int i = 0; i < sizeof(meyveler) / sizeof(meyveler[0]); i++)
    {
        cout << meyveler[i] << endl;
    }
    cout << "Liste disi index:" << meyveler[4] << endl;
    int a = 5;
    int &alias = a;
    a = 10;
    cout << alias << endl;
    string dosyaAdi = "ornek.txt"; // İşlem yapacağımız dosya adı

    // ---------------------------
    // 1️⃣ Dosyaya yazma (overwrite)
    // ---------------------------
    ofstream yazDosya(dosyaAdi); // ofstream: output file stream
    if (!yazDosya)
    { // Dosya açılamazsa kontrol
        cerr << "Dosya açılmadı!" << endl;
        return 1;
    }

    yazDosya << "Merhaba Dünya!\n";
    yazDosya << "C++ dosya işlemleri öğreniyorum.\n";
    yazDosya.close(); // Dosyayı kapatmayı unutma
    cout << "Dosyaya yazma işlemi tamamlandı." << endl;

    // ---------------------------
    // 2️⃣ Dosyadan okuma
    // ---------------------------
    ifstream okuDosya(dosyaAdi); // ifstream: input file stream
    if (!okuDosya)
    {
        cerr << "Dosya açılamadı!" << endl;
        return 1;
    }

    cout << "\nDosya içeriği:\n";
    string satir;
    while (getline(okuDosya, satir))
    { // Satır satır okuma
        cout << satir << endl;
    }
    okuDosya.close();

    // ---------------------------
    // 3️⃣ Dosyaya ekleme (append)
    // ---------------------------
    ofstream ekleDosya(dosyaAdi, ios::app); // ios::app -> ekleme modu
    if (!ekleDosya)
    {
        cerr << "Dosya açılamadı!" << endl;
        return 1;
    }

    ekleDosya << "Bu satır dosyanın sonuna eklendi.\n";
    ekleDosya.close();
    cout << "\nDosyaya ekleme işlemi tamamlandı." << endl;

    // ---------------------------
    // 4️⃣ Tekrar dosyayı oku
    // ---------------------------
    ifstream tekrarOku(dosyaAdi);
    cout << "\nGüncel dosya içeriği:\n";
    while (getline(tekrarOku, satir))
    {
        cout << satir << endl;
    }
    tekrarOku.close();
    ofstream yaz("dosya.txt");
    yaz << "merhabaa" << endl;
    yaz << "bu deneme" << endl;
    yaz.close();
    ifstream oku("dosya.txt");
    string line;
    while (getline(oku, line))
    {

        cout << line << endl;
    }
    oku.close();
    biri_bana_seslendi();

    selamVer("Bedo");

    string x[3] = {"Elma", "Armut", "Uzum"};
    sepetF(x);

    IleriHesaplama hesaplama;

    cout << hesaplama.topla(5, 4) << endl;
    cout << hesaplama.usAl(5, 2) << endl;
    cout << hesaplama.kokAl(25, 2) << endl;

    try
    {
        throw runtime_error("hata amk");
    }
    catch (const exception &e)
    {
        cout << "Sanki hata oldu gibi: " << e.what() << endl;
    }
    // kullanıcının girdiği sayıya kadar olan sayılardan çiftleri gösteren program
    int num;
    cout << "Bir sayi giriniz (cok yuksek olmasin):";
    cin >> num;
    // cin.ignore()

    for (int i = 0; i < num + 1; i++)
    {
        if (i % 2 == 0)
        {
            cout << "-->" << i << endl;
        }
        else
        {
            cout << i << endl;
        }
    }


    //Şifreli kapı
    string pass = "1234";
    string num;
    do{
        cout << "Sifreyi giriniz:";
        cin >> num;

    }while (num != pass);

    cout << "Sifre dogru";




    vector<int> liste = {1, 2, 3};

    // Sona ekleme
    liste.push_back(5); // liste: 1 2 3 5

    // Belirli bir indexe ekleme
    liste.insert(liste.begin() + 2, 10); // 2. indekse 10 ekle
    // liste: 1 2 10 3 5

    // Listeyi yazdır
    for (int i : liste)
        cout << i << " ";

    // Tek elemanı sil (2. indeks)
    liste.erase(liste.begin() + 2); // 3 silindi → liste: 1 2 4 5

    // Aralık silme (0. ve 1. indeks)
    liste.erase(liste.begin(), liste.begin() + 2); // 1 ve 2 silindi → liste: 4 5

    // Listeyi yazdır
    for (int i : liste)
        cout << i << " ";


    */
    std::cout << "Program başladı..." << std::endl;

    simdikiZaman();

    std::cout << "1 saniye bekleniyor..." << std::endl;
    bekle(1); // 1 saniye bekle

    simdikiZaman();

    std::cout << "Program bitti." << std::endl;

    return 0;
}
