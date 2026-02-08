import random
import math
import sqlite3
import sys
import re
import argparse
from helpers import helpers as hp, TokenizerMode, EMB_SIZE, EMB_RANGE, exp, log, tanh
import bytebpe as bpe



# =========================
# AYARLAR
# =========================
WINDOW = 2
EPOCHS = 100
LR = 0.1
NEG_SAMPLES = 5

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
    def __init__(
        self,
        activation_type="sigmoid",
        is_command_model=False,
        tokenizer_mode=TokenizerMode.WORD,
        subword_n=3,
        bpe_tokenizer=None,
        normalize_command_tokens=False
    ):
        self.vocab = {}
        self.W_in = {}
        self.W_out = {}
        self.vocab_list = []

        self.is_command_model = is_command_model
        self.tokenizer_mode = tokenizer_mode
        self.subword_n = subword_n
        self.bpe_tokenizer = bpe_tokenizer
        self.act_min, self.act_max, self.activation = hp.get_activation_info(activation_type) 
        self.activation_type = activation_type
        self.normalize_command_tokens = normalize_command_tokens

    def tokenize(
        self,
        sentence,
        mode=None,
        subword_n=None
    ):
        if mode is None:
            mode = self.tokenizer_mode
        if subword_n is None:
            subword_n = self.subword_n
        
        # helpers'den tokenize fonksiyonunu kullan
        return hp.tokenize(
            sentence,
            is_command=(self.is_command_model and self.normalize_command_tokens),
            mode=mode,
            subword_n=subword_n,
            bpe_tokenizer=self.bpe_tokenizer
        )


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

        vec = hp.normalize_vector(vec, self.act_min, self.act_max)
        return vec
    
    def normalize_all_embeddings(self):
        """
        Tüm kelime vektörlerini aktivasyon aralığına göre normalize eder
        """
        print(f"[INFO] Vektörler normalize ediliyor... [{self.act_min}, {self.act_max}]")
        
        for word in self.W_in:
            self.W_in[word] = hp.normalize_vector(self.W_in[word], self.act_min, self.act_max)
            self.W_out[word] = hp.normalize_vector(self.W_out[word], self.act_min, self.act_max)

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
            (word, ",".join(map(str, vec)), sentence_model.activation_type, 
             sentence_model.act_min, sentence_model.act_max)
        )

    conn.commit()
    conn.close()

def check_database(db_name):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
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
def load_file(filename, endThing=""):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [l.strip() + endThing for l in f if l.strip()]
        return lines
    except FileNotFoundError:
        print(f"[ERROR] '{filename}' dosyası bulunamadı!")
        return None



# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedding Oluşturucu - Aktivasyon Fonksiyonlu ve Parametre Tipleme Sistemli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s --activation relu --lr 0.15 --epochs 150
  %(prog)s -a tanh -l 0.2 -e 200 --tokenizer subword --subword-n 4
  %(prog)s --sentences data/sentences.txt --commands data/commands.txt
  %(prog)s --tokenizer bpe --bpe-vocab 2000
  %(prog)s --help

Mevcut Aktivasyon Fonksiyonları:
  sigmoid, tanh, relu, leaky_relu, elu, softplus, linear

Not: Komutlardaki parametreler otomatik olarak tip placeholderlarına dönüştürülür:
  mkdir test_folder  →  mkdir <DIR>
  rm file.txt        →  rm <FILE>
  cd /home/user      →  cd <PATH>
        """
    )
    
    # Aktivasyon ve hiperparametreler
    parser.add_argument(
        '-a', '--activation',
        type=str,
        default='sigmoid',
        choices=['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softplus', 'linear'],
        help='Aktivasyon fonksiyonu (default: sigmoid)'
    )
    
    parser.add_argument(
        '-l', '--lr',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=100,
        help='Epoch sayısı (default: 100)'
    )
    
    # Tokenizer ayarları
    parser.add_argument(
        '-t', '--tokenizer',
        type=str,
        default='word',
        choices=['word', 'subword', "bpe"],
        help='Tokenizer modu (default: word)'
    )
    
    parser.add_argument(
        '-n', '--subword-n',
        type=int,
        default=3,
        help='Subword n-gram değeri (default: 3)'
    )
    
    parser.add_argument(
        '--bpe-vocab',
        type=int,
        default=2000,
        help='BPE vocab boyutu (default: 2000)'
    )
    
    parser.add_argument(
        '--bpe-model',
        type=str,
        default='bpe_tokenizer.json',
        help='BPE model dosya adı (default: bpe_tokenizer.json)'
    )
    
    # Dosya yolları
    parser.add_argument(
        '-s', '--sentences',
        type=str,
        default='sentences.txt',
        help='Cümleler dosyası (default: sentences.txt)'
    )
    
    parser.add_argument(
        '-c', '--commands',
        type=str,
        default='commandVecs.txt',
        help='Komutlar dosyası (default: commandVecs.txt)'
    )
    
    parser.add_argument(
        '-d', '--db',
        type=str,
        default='embeddings.db',
        help='Cümleler için veritabanı adı (default: embeddings.db)'
    )
    
    parser.add_argument(
        '--db-commands',
        type=str,
        default='embeddingsForCommands.db',
        help='Komutlar için veritabanı adı (default: embeddingsForCommands.db)'
    )

    parser.add_argument(
        '--ctokenizer',
        type=str,
        default='False',
        help='Komutlar için cümleler ile aynı tokenizeri kullan'
    )

    parser.add_argument(
        '--normalize-command-tokens',
        action='store_true',
        help='Komutlardaki parametreleri tip placeholderlarına dönüştür'
    )

    
    args = parser.parse_args()
    
    # Banner yazdır
    print("""
╔═══════════════════════════════════╗
║     EMBEDDING OLUŞTURUCU (2 DOSYA)    ║
║     AKTİVASYON FONKSİYONLU SÜRÜM      ║
║  + PARAMETRE TİPLEME SİSTEMİ ✔       ║
║  + BPE TOKENIZER DESTEĞİ ✔           ║
╚═══════════════════════════════════╝
""")
    
    print(f"[AYARLAR]")
    print(f"  Aktivasyon: {args.activation}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Tokenizer: {args.tokenizer}")
    if args.tokenizer == 'subword':
        print(f"  Subword N: {args.subword_n}")
    if args.tokenizer == 'bpe':
        print(f"  BPE Vocab: {args.bpe_vocab}")
        print(f"  BPE Model: {args.bpe_model}")
    print(f"  Cümleler: {args.sentences}")
    print(f"  Komutlar: {args.commands}")
    print(f"  DB (Cümleler): {args.db}")
    print(f"  DB (Komutlar): {args.db_commands}")
    print(f"  ctokenizer   : {args.ctokenizer}")

    print()

    # Global değişkenleri güncelle
    LR = args.lr
    EPOCHS = args.epochs

    # Dosyaları yükle
    print(f"[INFO] Dosyalar okunuyor...")
    sentences = load_file(args.sentences)
    commands = load_file(args.commands, "")

    if sentences is None or commands is None:
        print("\n[ERROR] Dosyalar yüklenemedi!")
        print("\nÖrnek dosya formatları:")
        print(f"\n{args.sentences}:")
        print("  terminal aç")
        print("  dosya listele")
        print("  klasör oluştur")
        print(f"\n{args.commands}:")
        print("  bash")
        print("  ls")
        print("  mkdir")
        exit(1)

    if not sentences or not commands:
        print("[ERROR] Dosyalar boş!")
        exit(1)

    print(f"✓ {len(sentences)} cümle")
    print(f"✓ {len(commands)} komut")
    print(f"✓ Toplam {len(sentences) + len(commands)} satır\n")

    # BPE Tokenizer hazırlığı
    tokenizer = None
    if args.tokenizer == TokenizerMode.BPE:
        print(f"[BPE] Tokenizer oluşturuluyor (vocab_size={args.bpe_vocab})...")
        tokenizer = bpe.ByteBPETokenizer(vocab_size=args.bpe_vocab)

        # Tüm metni topla
        with open(args.sentences, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        words = [word for line in lines for word in line.split()]

        vocab_size = tokenizer.train(words)
        print(f"[BPE] Training vocab size: {vocab_size}")

        tokenizer.save(args.bpe_model)
        print(f"[BPE] Tokenizer kaydedildi: {args.bpe_model}")
        
        # Test
        print("\n[BPE] Test encoding:")
        for test_word in ["dizini", "dizine", "dizinine", "çık"][:3]:
            try:
                ids = tokenizer.encode(test_word)
                decoded = tokenizer.decode(ids)
                print(f"  '{test_word}' → {ids} → '{decoded}'")
            except:
                pass
        print()

    # Model oluştur
    print("[MODEL] Embedding modelleri oluşturuluyor...\n")
    
    sentence_model = SimpleWord2Vec(
        args.activation,
        is_command_model=False,
        tokenizer_mode=args.tokenizer,
        subword_n=args.subword_n,
        bpe_tokenizer=tokenizer
    )

    command_model = SimpleWord2Vec(
        args.activation,
        is_command_model=True,
        tokenizer_mode=TokenizerMode.WORD,
        subword_n=args.subword_n,
        bpe_tokenizer=None if args.ctokenizer == "False" else tokenizer, # Komutlar için BPE kullanmıyoruz
        normalize_command_tokens=args.normalize_command_tokens

    )

    print("[TRAIN] Eğitim başlıyor...\n")

    for i, line in enumerate(sentences, 1):
        tokens = sentence_model.add_sentence(line)
        
        # DEBUG: İlk 3 cümlenin token'larını göster
        if i <= 3:
            print(f"\n[DEBUG] Cümle {i}: '{line}'")
            print(f"[DEBUG] Token'lar: {tokens[:10]}...")  # İlk 10 token
        
        sentence_model.train_sentence(tokens)
        vec = sentence_model.sentence_embedding(line)

        bar = "█" * int(i / len(sentences) * 30)
        bar = bar.ljust(30, "-")
        mode_label = "BPE" if args.tokenizer == "bpe" else args.tokenizer.upper()
        print(f"\r[{bar}] {i}/{len(sentences)} [{mode_label}] | Vec[0]={vec[0]:.4f}", end="")
    
    print("\n[TRAIN] Komut embedding eğitimi (parametre tipleme aktif)...\n")

    for i, line in enumerate(commands, 1):
        tokens = command_model.add_sentence(line)
        command_model.train_sentence(tokens)
        vec = command_model.sentence_embedding(line)

        bar = "█" * int(i / len(commands) * 30)
        bar = bar.ljust(30, "-")
        print(f"\r[{bar}] {i}/{len(commands)} [CMD] | Vec[0]={vec[0]:.4f}", end="")
    print()

    # Tüm vektörleri normalize et
    sentence_model.normalize_all_embeddings()
    command_model.normalize_all_embeddings()

    # Veritabanına kaydet
    save_sqlite(sentence_model, args.db)
    save_sqlite(command_model, args.db_commands)

    print(f"\n\n✓ Eğitim tamamlandı → {args.db}")
    print(f"✓ Cümle kelime sayısı: {len(sentence_model.vocab)}")
    print(f"✓ Komut token sayısı: {len(command_model.vocab)}")
    
    print("\n[CHECK] Cümle Embeddings")
    check_database(args.db)

    print("\n[CHECK] Komut Embeddings (Tip Placeholderları)")
    check_database(args.db_commands)

    print("\n" + "="*50)
    print("ÖZET:")
    print(f"  Aktivasyon fonksiyonu: {sentence_model.activation_type}")
    print(f"  Çıktı aralığı: [{sentence_model.act_min}, {sentence_model.act_max}]")
    print(f"  Tokenizer: {args.tokenizer}")
    if args.tokenizer == 'bpe':
        print(f"  BPE vocab: {args.bpe_vocab}")
        print(f"  BPE model: {args.bpe_model}")
    print(f"  Cümleler: {len(sentences)}")
    print(f"  Komutlar: {len(commands)}")
    print(f"  Cümle token sayısı: {len(sentence_model.vocab)}")
    print(f"  Komut token sayısı: {len(command_model.vocab)}")
    print(f"  Veritabanı: {args.db}")
    print(f"  Veritabanı: {args.db_commands}")
    print("\n[ÖNEMLİ] Inference sırasında:")
    print("  1. Kullanıcı girdisini normalize edin (mkdir test → mkdir <DIR>)")
    print("  2. Embedding ile karşılaştırın")
    print("  3. En iyi komutu bulun")
    print("  4. Gerçek parametreleri geri enjekte edin")
    if args.tokenizer == 'bpe':
        print(f"  5. BPE tokenizer'ı {args.bpe_model} dosyasından yükleyin")
    print("="*50)