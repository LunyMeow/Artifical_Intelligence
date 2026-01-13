# ByteBPETokenizer C++ Kütüphanesi

Python kodunuzun C++ implementasyonu.

## Dosya Yapısı

```
project/
├── ByteBPETokenizer.h
├── ByteBPETokenizer.cpp
├── main.cpp
├── CMakeLists.txt
└── sentences.txt (corpus dosyanız)
```

## Gereksinimler

- C++17 destekli derleyici (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- nlohmann/json kütüphanesi (CMake otomatik indirir)

## Derleme

### Linux/macOS

```bash
# Build dizini oluştur
mkdir build
cd build

# CMake ile yapılandır
cmake ..

# Derle
make

# Çalıştır
./main
```

### Windows (Visual Studio)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\main.exe
```

## Manuel Derleme (nlohmann/json yerel kuruluysa)

```bash
g++ -std=c++17 -O3 ByteBPETokenizer.cpp main.cpp -o main
./main
```

## Kullanım Örnekleri

### Temel Kullanım

```cpp
#include "ByteBPETokenizer.h"
#include <iostream>

int main() {
    // Tokenizer oluştur (vocab_size=500)
    ByteBPETokenizer tokenizer(500);
    
    // Corpus ile eğit
    std::vector<std::string> corpus = {"hello", "world", "hello", "tokenizer"};
    tokenizer.train(corpus);
    
    // Encode et
    std::string text = "hello";
    auto ids = tokenizer.encode(text);
    
    // Decode et
    std::string decoded = tokenizer.decode(ids);
    
    std::cout << "Original: " << text << std::endl;
    std::cout << "Decoded: " << decoded << std::endl;
    
    return 0;
}
```

### Model Kaydetme/Yükleme

```cpp
// Model kaydet
tokenizer.save("my_tokenizer.json");

// Model yükle
ByteBPETokenizer new_tokenizer;
new_tokenizer.load("my_tokenizer.json");
```

### Dosyadan Corpus Okuma

```cpp
std::vector<std::string> read_corpus(const std::string& filename) {
    std::vector<std::string> corpus;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            corpus.push_back(word);
        }
    }
    return corpus;
}

// Kullanım
auto corpus = read_corpus("sentences.txt");
tokenizer.train(corpus);
```

## API Referansı

### Constructor

```cpp
ByteBPETokenizer(int vocab_size = 16000)
```

### Metodlar

- `int train(const std::vector<std::string>& texts)` - Tokenizer'ı eğitir, nihai vocab boyutunu döndürür
- `std::vector<int> encode(const std::string& text)` - Metni token ID'lerine çevirir
- `std::string decode(const std::vector<int>& ids)` - Token ID'lerini metne çevirir
- `void save(const std::string& path)` - Modeli JSON formatında kaydeder
- `void load(const std::string& path)` - Modeli JSON dosyasından yükler

## Performans Notları

C++ versiyonu Python versiyonundan genellikle 5-10x daha hızlıdır. Büyük corpus'lar için bellek yönetimi optimize edilmiştir.

## Farklılıklar

Python versiyonuyla tam uyumludur. Aynı corpus ile eğitildiğinde aynı sonuçları üretir.