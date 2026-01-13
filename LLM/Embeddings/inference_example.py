"""
INFERENCE EXAMPLE - Eğitilmiş modeli kullanma rehberi
=====================================================

Bu dosya, parametre tipleme sistemi ile eğitilmiş
embedding modelini DOĞRU ŞEKILDE kullanmayı gösterir.
"""

import sqlite3
import re
import math


# =========================
# AYARLAR
# =========================
EMB_SIZE = 50


# =========================
# PARAMETER NORMALIZATION (createEmbeddings.py ile AYNI)
# =========================
def normalize_command_params(tokens):
    """
    KRITIK: createEmbeddings.py'deki fonksiyonla TAMAMEN AYNI olmalı!
    """
    normalized = []
    
    for token in tokens:
        if token in ["<end>", "<start>"]:
            normalized.append(token)
            continue
        
        common_commands = [
            "mkdir", "rm", "cd", "ls", "cp", "mv", "touch", "cat",
            "grep", "find", "chmod", "chown", "sudo", "apt", "git",
            "python", "python3", "node", "npm", "bash", "sh", "echo",
            "tar", "zip", "unzip", "wget", "curl", "ssh", "scp"
        ]
        
        if token in common_commands:
            normalized.append(token)
            continue
        
        if token.startswith("-"):
            normalized.append(token)
            continue
        
        if re.match(r"^[/\\]|.*[/\\].*", token):
            normalized.append("<PATH>")
        elif re.match(r".*\.[a-zA-Z0-9]+$", token):
            normalized.append("<FILE>")
        elif re.match(r"^\d+$", token):
            normalized.append("<NUMBER>")
        elif re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", token):
            normalized.append("<IP>")
        elif re.match(r"^https?://", token):
            normalized.append("<URL>")
        elif re.match(r"^\$[A-Z_]+$", token):
            normalized.append("<VAR>")
        else:
            normalized.append("<DIR>")
    
    return normalized


# =========================
# EMBEDDING YÃœKLEME
# =========================
def load_embeddings(db_path):
    """Veritabanından embeddings yükle"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    embeddings = {}
    cur.execute("SELECT word, vector FROM embeddings")
    
    for word, vec_str in cur.fetchall():
        vec = [float(x) for x in vec_str.split(",")]
        if len(vec) == EMB_SIZE:
            embeddings[word] = vec
    
    conn.close()
    return embeddings

def load_lines(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Dosya bulunamadı: {filename}")
        return []



# =========================
# SENTENCE EMBEDDING
# =========================
def sentence_embedding(sentence, embeddings, normalize_params=False):
    """
    Cümle embedding hesapla
    
    Args:
        sentence: Girdi metni
        embeddings: Kelime vektörleri
        normalize_params: True ise parametre normalizasyonu uygula (komutlar için)
    """
    tokens = sentence.lower().split()
    
    if normalize_params:
        tokens = normalize_command_params(tokens)
    
    vec = [0.0] * EMB_SIZE
    count = 0
    
    for word in tokens:
        if word in embeddings:
            for i in range(EMB_SIZE):
                vec[i] += embeddings[word][i]
            count += 1
    
    if count > 0:
        for i in range(EMB_SIZE):
            vec[i] /= count
    
    return vec, tokens


# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(vec1, vec2):
    """İki vektör arasındaki kosinüs benzerliği"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


# =========================
# KOMUT TAHMÄ°N MOTORU
# =========================
class CommandPredictor:
    def __init__(self, sentence_db="embeddings.db", command_db="embeddingsForCommands.db"):
        print("[LOAD] Sentence embeddings yükleniyor...")
        self.sentence_embeddings = load_embeddings(sentence_db)
        
        print("[LOAD] Command embeddings yükleniyor...")
        self.command_embeddings = load_embeddings(command_db)
        
        print(f"✓ {len(self.sentence_embeddings)} sentence tokens")
        print(f"✓ {len(self.command_embeddings)} command tokens\n")
    
    def predict_command(self, user_input, known_commands):
        """
        Kullanıcı girdisine en uygun komutu bul
        
        Args:
            user_input: Kullanıcının doğal dil cümlesi
            known_commands: Bilinen komutların listesi (parametreli)
        
        Returns:
            (best_command, similarity_score, normalized_input)
        """
        # 1. Kullanıcı girdisini embedding'e çevir (PARAMETRE NORMALIZASYONU YOK)
        input_vec, _ = sentence_embedding(user_input, self.sentence_embeddings, normalize_params=False)
        
        best_match = None
        best_score = -1
        best_normalized = None
        
        # 2. Her bilinen komutu kontrol et
        for cmd in known_commands:
            # Komutu normalize et (PARAMETRE NORMALIZASYONU VAR)
            cmd_vec, normalized_tokens = sentence_embedding(
                cmd, 
                self.command_embeddings, 
                normalize_params=True
            )
            
            # 3. Benzerlik hesapla
            score = cosine_similarity(input_vec, cmd_vec)
            
            if score > best_score:
                best_score = score
                best_match = cmd
                best_normalized = " ".join(normalized_tokens)
        
        return best_match, best_score, best_normalized


# =========================
# KULLANIM ÖRNEKLERİ
# =========================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════╗
║      INFERENCE EXAMPLE SCRIPT        ║
║   Parametre Tipleme ile Komut Tahmin  ║
╚═══════════════════════════════════════╝
""")
    
    # Predictor oluştur
    predictor = CommandPredictor()
    
    # Bilinen komutlar (gerçek parametrelerle)
    # =========================
    # TEST VERİLERİNİ DOSYADAN OKU
    # =========================
    
    sentences_file = "sentences.txt"
    commands_file = "commandVecs.txt"
    
    print("[LOAD] Test cümleleri yükleniyor...")
    test_inputs = load_lines(sentences_file)
    
    print("[LOAD] Bilinen komutlar yükleniyor...")
    known_commands = load_lines(commands_file)
    
    if not test_inputs:
        print("[ERROR] sentences.txt boş veya okunamadı")
        exit(1)
    
    if not known_commands:
        print("[ERROR] commandVecs.txt boş veya okunamadı")
        exit(1)
    
    print("\n[INFO] Bilinen komutlar:")
    for cmd in known_commands:
        print(f"  - {cmd}")
    
    print("\n" + "=" * 50)
    print("TEST CASELER")
    print("=" * 50 + "\n")

    
    for user_input in test_inputs:
        print(f"Kullanıcı: '{user_input}'")
        
        best_cmd, score, normalized = predictor.predict_command(user_input, known_commands)
        
        print(f"  → En iyi eşleşme: {best_cmd}")
        print(f"  → Benzerlik skoru: {score:.4f}")
        print(f"  → Normalize form: {normalized}")
        print()
    
    print("="*50)
    print("ÖNEMLİ NOTLAR:")
    print("="*50)
    print("""
1. KULLANICI GİRDİSİ: Doğal dil, parametre normalizasyonu YOK
   - "yeni klasör oluştur" → doğrudan embedding

2. KOMUT KARŞILAŞTIRMA: Parametreler tip placeholderlarına dönüşür
   - "mkdir new_folder" → "mkdir <DIR>"
   - Bu sayede "mkdir test", "mkdir project" hepsi aynı semantiğe sahip

3. INFERENCE WORKFLOW:
   ┌─────────────────────────────────────────┐
   │ User Input: "klasör oluştur"           │
   └──────────────┬──────────────────────────┘
                  │ (parametre norm YOK)
                  v
   ┌─────────────────────────────────────────┐
   │ Sentence Embedding                      │
   └──────────────┬──────────────────────────┘
                  │
                  v
   ┌─────────────────────────────────────────┐
   │ Known Commands:                         │
   │   "mkdir test"  → "mkdir <DIR>"        │
   │   "rm file.txt" → "rm <FILE>"          │
   └──────────────┬──────────────────────────┘
                  │ (parametre norm VAR)
                  v
   ┌─────────────────────────────────────────┐
   │ Command Embeddings                      │
   └──────────────┬──────────────────────────┘
                  │
                  v
   ┌─────────────────────────────────────────┐
   │ Cosine Similarity                       │
   └──────────────┬──────────────────────────┘
                  │
                  v
   ┌─────────────────────────────────────────┐
   │ Best Match: "mkdir test"                │
   │ (Gerçek parametrelerle döndür)         │
   └─────────────────────────────────────────┘

4. PARAMETRE ENJEKSİYONU:
   - Tahmin edilen komut: "mkdir <DIR>"
   - Kullanıcı yeni parametre verebilir: "project_folder"
   - Son komut: "mkdir project_folder"

Bu sistem sayede:
  ✓ "mkdir test", "mkdir project", "mkdir abc" → hepsi aynı semantik
  ✓ Embedding uzayı bozulmaz
  ✓ Yeni parametre değerleri eklenebilir
  ✓ Komut mantığı korunur
""")
    
    print("\n[İNTERAKTİF MOD]")
    print("Çıkmak için 'exit' yazın\n")
    
    while True:
        user_input = input("Komut isteği: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Çıkış yapılıyor...")
            break
        
        if not user_input:
            continue
        
        best_cmd, score, normalized = predictor.predict_command(user_input, known_commands)
        
        print(f"\n  ✓ Tahmin: {best_cmd}")
        print(f"  ✓ Skor: {score:.4f}")
        print(f"  ✓ Normalize: {normalized}\n")