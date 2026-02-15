# ✅ PROJE BİLGİ VE YAPILACAKLAR

## 📚 Ana Dokümantasyon

Tüm proje bilgileri **tek bir dosyada** birleştirilmiştir:

**📖 FILE: `PROJECT_MASTER_GUIDE.md`**
- ✅ Proje genel bakışı
- ✅ Kurulum ve derleme talimatları
- ✅ Veri hazırlama ve training
- ✅ Terminal LLM sistemi (yeni)
- ✅ Web deployment (Express.js + WASM)
- ✅ API referansı (C++, JavaScript, Python)
- ✅ Sorun çözme rehberi
- ✅ Performans metrikleri
- ✅ Dosya envanteri
- ✅ Hızlı referans komutları

**Ne yapılır:**
- Proje'nin nasıl çalıştığı
- Derleme adımları
- Veri nasıl hazırlanır
- Training nasıl yapılır
- Web nasıl deploy edilir
- Tüm komutlar
- Tüm dosyaların açıklaması

---

## 🗑️ SİLİNEBİLECEK GERESİZ DOSYALAR

Tüm gereksiz dosyalar **ayrıntılı olarak listelenmiştir**:

**📄 FILE: `UNNECESSARY_FILES_TO_DELETE.txt`**

Bu dosyada liste:

### ❌ HEMEN SİLİNEBİLİR (13 dosya)
Çünkü PROJECT_MASTER_GUIDE.md'de birleştirildi:

1. COMMAND_PARAM_EXTRACTOR_README.md
2. TOKEN_GENERATION_FIXES.md
3. TERMINAL_LLM_INTEGRATION.md
4. README_TERMINAL_LLM.md
5. README_INDEX.md
6. QUICK_REFERENCE.md
7. COMPILATION_REPORT.md
8. COMPLETION_NOTICE.txt
9. PROJECT_COMPLETION_SUMMARY.md
10. PROJECT_STATUS.txt
11. INFERENCE_ENGINE_INTEGRATION.md
12. TESTING_CHECKLIST.md
13. CHANGELOG.md

### ⚠️ SİLİNMESİ ÖNERİLEN (20+ dosya)
Eski versiyon dosyaları:

- Buildv1_0_0.py (eski)
- Buildv1_1_1.cpp (eski)
- Buildv1_3_1.cpp (eski)
- buildController.py (eski)
- modeltrainingprogram.py (eski)
- NeuronAndConnection.py (eski)
- neuronv*.py (tümü - 10 dosya)
- torchExamples.py
- anothermodelfromanai.py
- build (derlenmiş binary)
- test_param_extractor (test binary)

### 👍 KORUNMALI DOSYALAR
Asla silmeyin:

- Buildv1_3_2.cpp (CURRENT)
- ByteBPE/ (Tokenizer)
- LLM/Embeddings/ (Veri ve Python scripts)
- web/ (Web sunucusu)
- ParameterExtractorV2.h/cpp (Parametre çıkarım)
- InferenceEngine.h/cpp (Token generation)
- nativeMaker.sh, wasmMaker.sh (Build scriptleri)

---

## 🚀 PROJE AYAKTA KALMAK İÇİN GEREKLİ

### ✅ Tüm Bilgiler PROJECT_MASTER_GUIDE.md'de

```bash
# KURULUM
g++ -o build Buildv1_3_2.cpp ByteBPE/ByteBPETokenizer.cpp -std=c++17 -lsqlite3

# ÇALIŞTIRMA
./build

# WEB
cd web && npm install && npm start

# VERİ HAZIRLA
cd LLM/Embeddings
python3 generate_template_training.py
```

### ✅ Terminal LLM Kullanalım

```bash
./build
> generate backup dosyasını projelere kopyala
```

Output:
```
Command: cp
Parameters: <SRC>=backup, <DST>=projeler
Generated: cp backup projeler <end>
```

---

## 📋 SİLME PROSEDÜRÜ

```bash
cd /home/kali/Desktop/Projects/Artifical_Intelligence

# 1. Tüm eski MD dosyaları sil
rm COMMAND_PARAM_EXTRACTOR_README.md
rm TOKEN_GENERATION_FIXES.md
rm TERMINAL_LLM_INTEGRATION.md
rm README_TERMINAL_LLM.md
rm README_INDEX.md
rm QUICK_REFERENCE.md
rm COMPILATION_REPORT.md
rm COMPLETION_NOTICE.txt
rm PROJECT_COMPLETION_SUMMARY.md
rm PROJECT_STATUS.txt
rm INFERENCE_ENGINE_INTEGRATION.md
rm TESTING_CHECKLIST.md
rm CHANGELOG.md

# 2. Eski version dosyaları sil
rm Buildv1_0_0.py
rm Buildv1_1_1.cpp
rm Buildv1_3_1.cpp
rm buildController.py
rm modeltrainingprogram.py
rm NeuronAndConnection.py
rm anothermodelfromanai.py
rm neuronv*.py
rm torchExamples.py

# 3. Build artifacts (opsiyonel - rebuild edilebilir)
rm build
rm test_param_extractor

# Toplam silinecek boyut: ~150+ MB
```

---

## ✨ PROJENIN DURUMU

**Sürüm:** 1.3.2 (Production Ready)  
**Status:** ✅ Tam Fonksiyonel  

### Tamamlanan Özellikler
- ✅ Neural Network (Feedforward + Training)
- ✅ Dinamik Mimari Optimizasyonu
- ✅ BPE Tokenization
- ✅ Embedding Sistemi
- ✅ Native C++ Binary
- ✅ WebAssembly Desteği
- ✅ Web Sunucusu
- ✅ Parameter Extraction (7 tip)
- ✅ Terminal LLM Sistemi
- ✅ Çok Dillililik (English + Turkish)
- ✅ Şablon Tabanlı Komut Üretimi

### Son Hatalar Düzeltildi
- ✅ InferenceEngine dangling pointers
- ✅ argv[2] null dereference
- ✅ WASM mode initialization
- ✅ CLI mode initialization

---

## 📞 HIZLI REFERANSİ

| İşlem | Komut |
|-------|-------|
| Derle | `bash nativeMaker.sh` |
| Çalıştır | `./build` |
| Web başlat | `cd web && npm start` |
| WASM derle | `bash wasmMaker.sh` |
| Veri hazırla | `python3 LLM/Embeddings/generate_template_training.py` |
| Embedding eğit | `python3 LLM/Embeddings/train_enhanced_embeddings.py` |
| Sorunu gider | Bkz: PROJECT_MASTER_GUIDE.md Troubleshooting |

---

## 📖 DÖKÜMAN OKUMA SIRASI

1. **Başla:** PROJECT_MASTER_GUIDE.md (Ana Rehber)
2. **Kurulum:** "INSTALLATION & SETUP" bölümü
3. **Veri:** "DATA PREPARATION" bölümü
4. **Training:** "TRAINING & CONFIGURATION" bölümü
5. **Terminal LLM:** "TERMINAL LLM SYSTEM" bölümü
6. **Web:** "WEB DEPLOYMENT" bölümü
7. **Sorunlar:** "TROUBLESHOOTING" bölümü

---

**Hazırlanma Tarihi:** 11 Şubat 2026  
**Durum:** ✅ Tamamlandı  
**Sonraki Adım:** Eski dosyaları UNNECESSARY_FILES_TO_DELETE.txt'deki listeden sil
