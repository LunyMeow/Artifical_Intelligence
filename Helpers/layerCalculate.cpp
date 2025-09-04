#include <iostream>
#include <cmath>
using namespace std;

// 1. Basit ortalama kurali
int hiddenLayers_average(int inputSize, int outputSize) {
    int avg = (inputSize + outputSize) / 2;
    if (avg < 10) return 1;
    else if (avg < 50) return 2;
    else return 3;
}

// 2. Logaritmik yaklasim
int hiddenLayers_logImproved(int inputSize, int outputSize, const string& level) {
    double ratio = (double)inputSize / (double)outputSize;
    double complexity = log2(ratio + 1) + log2(inputSize + outputSize);

    // Temel katman sayısı
    int layers = (int)round(complexity / 2.0);

    // Seviye ayarı
    double multiplier = 1.0;
    if (level == "low") multiplier = 0.6;
    else if (level == "medium") multiplier = 1.0;
    else if (level == "high") multiplier = 1.5;

    layers = (int)round(layers * multiplier);

    // 1–12 arası sınırla
    if (layers < 1) layers = 1;
    if (layers > 13) layers = 12;

    return layers;
}



// 3. Karmasik problem yaklasimi
int hiddenLayers_complexity(int inputSize, int outputSize) {
    if (inputSize <= 5 || outputSize <= 2) return 1;
    if (inputSize > 100 && outputSize < 10) return 4;
    if (inputSize > 50) return 3;
    return 2;
}

// 4. Sabit temel kurali
int hiddenLayers_fixed(int inputSize, int outputSize) {
    return 2; // her zaman 2 gizli katman oner
}

int main() {
    int inputSize = 64;  // ornek giris
    int outputSize = 10; // ornek cikis

    cout << "Input giriniz:" ;
    cin >> inputSize;
    cout << "Out giriniz:";
    cin >> outputSize;

    string level = "";
    cout << "Level giriniz:";
    cin >> level;

    cout << "Basit Ortalama Kurali: " << hiddenLayers_average(inputSize, outputSize) << " gizli katman" << endl;
    cout << "Logaritmik Yaklasim: " << hiddenLayers_logImproved(inputSize, outputSize,level) << " gizli katman" << endl;
    cout << "Karmasiklik Yaklasimi: " << hiddenLayers_complexity(inputSize, outputSize) << " gizli katman" << endl;
    cout << "Sabit Kural: " << hiddenLayers_fixed(inputSize, outputSize) << " gizli katman" << endl;

    return 0;
}
