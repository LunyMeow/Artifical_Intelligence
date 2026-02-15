# InferenceEngine + Command Schema JSON Integration

## Özet

InferenceEngine'in `command_schema.json` dosyasını dinamik olarak yükleyerek, daha kesin ve esnek bir parametre çıkarma sistemi kurulmuştur.

---

## Değişiklikler

### 1. **InferenceEngine.h** 
- `#include <nlohmann/json.hpp>` eklendi
- Yeni private method: `load_command_schema(const std::string &schema_path)`
- Yeni public method: `bool init_with_schema(const std::string &schema_path)`
- New member: `std::unordered_map<std::string, std::string> command_descriptions_` (metadata)

### 2. **InferenceEngine.cpp**
- JSON parsing ve yükleme fonksiyonları eklendi
- `load_command_schema()`: JSON dosyasını okur ve template'leri dinamik olarak oluşturur
- `init_with_schema()`: Public API - JSON şemasıyla başlatma
- Fallback: JSON yükleme başarısız olursa `init_templates()` kullanılır

### 3. **Buildv1_3_2.cpp**
- `load_user_model()` fonksiyonunda `g_inference_engine->init_with_schema()` çağrısı eklendi
- Şema yolu: `"LLM/Embeddings/cmdparam/command_schema.json"`

---

## Avantajlar

### ✅ Dinamiklik
- Yeni komutlar JSON'a eklenerek otomatik yüklenir
- Program recompile etmeye gerek yok

### ✅ Kesinlik
- JSON'da her komutun parametreleri açıkça tanımlanır
- `<SRC>`, `<DST>`, `<FILE>` gibi tip tanımlarına göre parametre seçilir

### ✅ Açıklama Desteği
- Her komut için Türkçe açıklama (future: kullanıcıya gösterilebilir)
- `command_descriptions_` map'inde saklanır

### ✅ Ölçeklenebilirlik
- 27 komut şemasından yükle
- 100+ komut eklemek kolay

---

## JSON Şema Formatı

```json
{
  "cp": {
    "params": ["<src>", "<dst>"],
    "description": "Dosya veya klasörü kopyalar"
  },
  "chmod": {
    "params": ["<mode>", "<file>"],
    "description": "Dosya izinlerini değiştirir"
  }
}
```

---

## Yüklenen Komutlar (27 toplam)

- **Dizin işlemleri**: cd, ls, pwd, mkdir, rmdir, find
- **Dosya işlemleri**: touch, rm, cp, mv, cat, less, nano, vim, grep
- **İzin işlemleri**: chmod, chown
- **Arşiv işlemleri**: tar, zip, unzip
- **İnternet**: wget, curl
- **Sistem**: ps, kill, top
- **Diğer**: clear, exit

---

## Kullanım

### Otomatik (Application Startup)
```cpp
// Buildv1_3_2.cpp'de otomatik olarak çağrılır:
g_inference_engine->init_with_schema("LLM/Embeddings/cmdparam/command_schema.json");
```

### Manual
```cpp
InferenceEngine engine(extractor, word_emb, cmd_emb, debug);
bool success = engine.init_with_schema("path/to/command_schema.json");
```

---

## Test Sonuçları

```
[load_command_schema] Successfully loaded 27 commands
[init_with_schema] Total commands available: 27
[SUCCESS] Command schema loaded from JSON!
```

---

## Parametre Çıkarma Akışı

```
1. Kullanıcı: "backup klasörüne yedekleri kopyala"
2. Tahmin: "cp" komutu
3. Template lookup: cp → ["<src>", "<dst>"]
4. extract_parameters_from_sentence():
   - Position-based: token 0 → <src>, token 1 → <dst>
   - Heuristic scoring: src/file tipi, dst/dir tipi
5. Result: "cp backup klasörüne"
```

---

## Fallback Mekanizması

JSON dosyası bulunamazsa:
- `init_templates()` çağrılır (hardcoded 23 komut)
- Sistem normal çalışmaya devam eder
- Debug log: "Using fallback command templates"

---

## Future Improvements

1. **Multi-language support**: JSON'a dil desteği ekle
2. **Parameter validation**: Parametre tipleri doğrula
3. **Command aliases**: "kopyala" → "cp" mapping
4. **Priority ordering**: En sık kullanılan komutlar önce
5. **Dynamic reloading**: Runtime'da JSON reload

---

## Files Modified

- ✅ `InferenceEngine.h` - Header güncellemesi
- ✅ `InferenceEngine.cpp` - JSON loading implementation
- ✅ `Buildv1_3_2.cpp` - Schema initialization
- ✅ `test_inference.cpp` - JSON testing

## Files Used

- 📄 `LLM/Embeddings/cmdparam/command_schema.json` - Command definitions (27 komut)

---

## Compile Komutu

```bash
g++ -std=c++17 -O2 -I./ByteBPE -I./include \
    -o Buildv1_3_2 \
    Buildv1_3_2.cpp InferenceEngine.cpp \
    CommandParamExtractor.cpp ByteBPE/ByteBPETokenizer.cpp \
    -lsqlite3
```

✅ **Başarılı!** JSON desteği tamamlandı.
