import random
import math
import sqlite3
import sys


# =========================
# AYARLAR
# =========================
EMB_SIZE = 50
WINDOW = 2
EPOCHS = 100
LR = 0.1
NEG_SAMPLES = 5

EMB_RANGE = range(EMB_SIZE)

# =========================
# AKTİVASYON FONKSİYONLARI
# =========================
exp = math.exp
log = math.log
tanh = math.tanh

def get_activation_info(activation_type):
    """
    Aktivasyon fonksiyonunun çıktı aralığını döndürür
    Returns: (min_val, max_val, activation_function)
    """
    activations = {
        "sigmoid": (0.0, 1.0, lambda x: 1.0 / (1.0 + exp(-x))),
        "tanh": (-1.0, 1.0, lambda x: tanh(x)),
        "relu": (0.0, float('inf'), lambda x: max(0.0, x)),
        "leaky_relu": (float('-inf'), float('inf'), lambda x: x if x > 0.0 else 0.01 * x),
        "elu": (-0.01, float('inf'), lambda x: x if x >= 0.0 else 0.01 * (exp(x) - 1.0)),
        "softplus": (0.0, float('inf'), lambda x: log(1.0 + exp(x))),
        "linear": (float('-inf'), float('inf'), lambda x: x)
    }
    
    return activations.get(activation_type, activations["sigmoid"])

def normalize_vector(vec, min_val, max_val):
    """
    Vektörü aktivasyon fonksiyonunun çıktı aralığına göre normalize eder
    """
    # Vektörün mevcut min-max değerlerini bul
    vec_min = min(vec)
    vec_max = max(vec)
    
    if vec_max == vec_min:
        # Tüm değerler aynıysa, aralığın ortasını kullan
        normalized = [(min_val + max_val) / 2.0] * len(vec)
    else:
        # Min-max normalizasyonu (0-1 aralığına)
        normalized = [(v - vec_min) / (vec_max - vec_min) for v in vec]
        
        # Hedef aralığa ölçeklendir
        if min_val != float('-inf') and max_val != float('inf'):
            normalized = [min_val + v * (max_val - min_val) for v in normalized]
        elif max_val != float('inf'):
            # Sadece üst sınır var (relu, softplus gibi)
            normalized = [min_val + v * max_val for v in normalized]
        elif min_val != float('-inf'):
            # Sadece alt sınır var
            normalized = [min_val + v * abs(min_val) for v in normalized]
    
    return normalized

# =========================
# YARDIMCI
# =========================
def dot(a, b):
    s = 0.0
    for i in EMB_RANGE:
        s += a[i] * b[i]
    return s

def random_vector():
    return [random.uniform(-0.5, 0.5) for _ in EMB_RANGE]

# =========================
# MODEL
# =========================
class SimpleWord2Vec:
    def __init__(self, activation_type="sigmoid"):
        self.vocab = {}
        self.W_in = {}
        self.W_out = {}
        self.vocab_list = []
        
        # Aktivasyon fonksiyonu bilgilerini al
        self.act_min, self.act_max, self.activation = get_activation_info(activation_type)
        self.activation_type = activation_type
        print(f"[INFO] Aktivasyon fonksiyonu: {activation_type}")
        print(f"[INFO] Çıktı aralığı: [{self.act_min}, {self.act_max}]")

    def tokenize(self, sentence):
        return sentence.lower().split()

    def add_sentence(self, sentence):
        tokens = self.tokenize(sentence)

        for w in tokens:
            if w not in self.vocab:
                self.vocab[w] = 1
                self.W_in[w] = random_vector()
                self.W_out[w] = random_vector()
                self.vocab_list.append(w)

        return tokens

    def train_sentence(self, sent):
        W_in = self.W_in
        W_out = self.W_out
        vocab_list = self.vocab_list
        activation = self.activation

        for _ in range(EPOCHS):
            L = len(sent)
            for i in range(L):
                w = sent[i]
                vin = W_in[w]

                start = max(0, i - WINDOW)
                end = min(L, i + WINDOW + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    # POSITIVE
                    w_out = sent[j]
                    vout = W_out[w_out]

                    score = dot(vin, vout)
                    # Aktivasyon fonksiyonunu kullan
                    activated_score = activation(score)
                    error = 1.0 - activated_score
                    grad = LR * error

                    for k in EMB_RANGE:
                        tmp = vin[k]
                        vin[k] += grad * vout[k]
                        vout[k] += grad * tmp

                    # NEGATIVE
                    for _ in range(NEG_SAMPLES):
                        neg = random.choice(vocab_list)
                        vneg = W_out[neg]

                        score = dot(vin, vneg)
                        # Aktivasyon fonksiyonunu kullan
                        activated_score = activation(score)
                        error = -activated_score
                        grad = LR * error

                        for k in EMB_RANGE:
                            tmp = vin[k]
                            vin[k] += grad * vneg[k]
                            vneg[k] += grad * tmp

    def sentence_embedding(self, sentence):
        tokens = self.tokenize(sentence)
        vec = [0.0] * EMB_SIZE
        cnt = 0

        W_in = self.W_in

        for w in tokens:
            if w in W_in:
                vw = W_in[w]
                for i in EMB_RANGE:
                    vec[i] += vw[i]
                cnt += 1

        if cnt:
            inv = 1.0 / cnt
            for i in EMB_RANGE:
                vec[i] *= inv

        # Aktivasyon aralığına göre normalize et
        vec = normalize_vector(vec, self.act_min, self.act_max)

        return vec
    
    def normalize_all_embeddings(self):
        """
        Tüm kelime vektörlerini aktivasyon aralığına göre normalize eder
        """
        print(f"[INFO] Vektörler normalize ediliyor... [{self.act_min}, {self.act_max}]")
        
        for word in self.W_in:
            self.W_in[word] = normalize_vector(self.W_in[word], self.act_min, self.act_max)
            self.W_out[word] = normalize_vector(self.W_out[word], self.act_min, self.act_max)

# =========================
# KAYIT
# =========================
def save_sqlite(sentence_model, db_name="embeddings.db"):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        word TEXT PRIMARY KEY,
        vector TEXT,
        activation_type TEXT,
        act_min REAL,
        act_max REAL
    )
    """)

    for word, vec in sentence_model.W_in.items():
        cur.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
            (word, ",".join(map(str, vec)), sentence_model.activation_type, sentence_model.act_min, sentence_model.act_max)
        )

    conn.commit()
    conn.close()





def check_database(db_name):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    # Get schema
    cur.execute("PRAGMA table_info(embeddings)")
    schema = cur.fetchall()
    print("Database Schema:")
    for col in schema:
        print(f"  {col[1]}: {col[2]}")
    
    print("\nSample embeddings:")
    cur.execute("SELECT word, vector, activation_type, act_min, act_max FROM embeddings LIMIT 5")
    for row in cur.fetchall():
        word, vector, act_type, act_min, act_max = row
        vec_values = vector.split(',')
        print(f"\nWord: {word}")
        print(f"  Activation: {act_type} [{act_min}, {act_max}]")
        print(f"  Vector (first 5 dims): {vec_values[:5]}")
    
    print(f"\nTotal words: {cur.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]}")
    
    conn.close()



# =========================
# DOSYA OKUMA
# =========================
def load_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines
    except FileNotFoundError:
        print(f"[ERROR] '{filename}' dosyası bulunamadı!")
        return None

# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════╗
║     EMBEDDING OLUŞTURUCU (2 DOSYA)    ║
║     AKTİVASYON FONKSİYONLU SÜRÜM      ║
╚═══════════════════════════════════════╝
""")





    # Aktivasyon fonksiyonunu kontrol et
    available_activations = ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "softplus", "linear"]
    
    if len(sys.argv) > 1:
        activation_type = sys.argv[1].lower()
        if activation_type not in available_activations:
            print(f"[ERROR] Geçersiz aktivasyon fonksiyonu: {activation_type}")
            print(f"[INFO] Mevcut seçenekler: {', '.join(available_activations)}")
            activation_type = input("Aktivasyon fonksiyonu seçin [sigmoid]: ").strip().lower() or "sigmoid"
    else:
        print(f"[INFO] Mevcut aktivasyon fonksiyonları: {', '.join(available_activations)}")
        activation_type = input("Aktivasyon fonksiyonu seçin [sigmoid]: ").strip().lower() or "sigmoid"
    
    print(f"\n[INFO] Seçilen aktivasyon: {activation_type}\n")

    # Dosya adlarını al
    sentences_file = input("Cümleler dosyası [sentences.txt]: ").strip() or "sentences.txt"
    commands_file = input("Komutlar dosyası [commandVecs.txt]: ").strip() or "commandVecs.txt"
    db_file = input("Veritabanı adı [embeddings.db]: ").strip() or "embeddings.db"
    db_file_commands = input("Komutlar için veritabanı adı [embeddingsForCommands.db]: ").strip() or "embeddingsForCommands.db"

    # Dosyaları yükle
    print(f"\n[INFO] Dosyalar okunuyor...")
    sentences = load_file(sentences_file)
    commands = load_file(commands_file)

    if sentences is None or commands is None:
        print("\n[ERROR] Dosyalar yüklenemedi!")
        print("\nÖrnek dosya formatları:")
        print(f"\n{sentences_file}:")
        print("  terminal aç")
        print("  dosya listele")
        print("  klasör oluştur")
        print(f"\n{commands_file}:")
        print("  bash")
        print("  ls")
        print("  mkdir")
        exit(1)

    if not sentences:
        print(f"[ERROR] '{sentences_file}' boş!")
        exit(1)

    if not commands:
        print(f"[ERROR] '{commands_file}' boş!")
        exit(1)

    # Tüm cümleleri birleştir
    all_lines = sentences + commands


    print(f"✓ {len(sentences)} cümle")
    print(f"✓ {len(commands)} komut")
    print(f"✓ Toplam {len(all_lines)} satır\n")

    # Model oluştur ve eğit
    sentence_model = SimpleWord2Vec(activation_type)
    command_model = SimpleWord2Vec(activation_type)

    print("[TRAIN] Eğitim başlıyor...\n")

    for i, line in enumerate(sentences, 1):
        tokens = sentence_model.add_sentence(line)
        sentence_model.train_sentence(tokens)

        vec = sentence_model.sentence_embedding(line)

        bar = "█" * int(i / len(sentences) * 30)
        bar = bar.ljust(30, "-")

        # Hangi dosyadan olduğunu göster
        source = "CMD" 
        print(f"\r[{bar}] {i}/{len(sentences)} [{source}] | Vec[0]={vec[0]:.4f}", end="")
    
    # -------- KOMUTLAR --------
    print("\n[TRAIN] Komut embedding eğitimi...\n")

    for i, line in enumerate(commands, 1):
        tokens = command_model.add_sentence(line)
        command_model.train_sentence(tokens)

        vec = command_model.sentence_embedding(line)

        bar = "█" * int(i / len(commands) * 30)

        bar = bar.ljust(30, "-")

        # Hangi dosyadan olduğunu göster
        source = "CMD" 
        print(f"\r[{bar}] {i}/{len(commands)} [{source}] | Vec[0]={vec[0]:.4f}", end="")

    # Tüm vektörleri normalize et
    sentence_model.normalize_all_embeddings()
    command_model.normalize_all_embeddings()


    # Veritabanına kaydet
    save_sqlite(sentence_model, db_file)
    save_sqlite(command_model, db_file_commands)

    print(f"\n\n✓ Eğitim tamamlandı → {db_file}")
    print(f"✓ Toplam {len(sentence_model.vocab)} kelime öğrenildi")
    
    
    print("\n[CHECK] Cümle Embeddings")
    check_database(db_file)

    print("\n[CHECK] Komut Embeddings")
    check_database(db_file_commands)

    # Özet bilgi
    print("\n" + "="*50)
    print("ÖZET:")
    print(f"  Aktivasyon fonksiyonu: {sentence_model.activation_type}")
    print(f"  Çıktı aralığı: [{sentence_model.act_min}, {sentence_model.act_max}]")
    print(f"  Cümleler: {len(sentences)}")
    print(f"  Komutlar: {len(commands)}")
    print(f"  Cümle kelime sayısı: {len(sentence_model.vocab)}")
    print(f"  Komut kelime sayısı: {len(command_model.vocab)}")

    print(f"  Veritabanı: {db_file}")
    print("="*50)