"""
createEmbeddings.py  —  Unified Embedding Eğiticisi
=====================================================
DEĞİŞİKLİK ÖZETİ (eski → yeni):
  - İki ayrı DB (embeddings.db + embeddingsForCommands.db) KALDIRILDI.
  - Tek bir DB (embeddings.db) oluşturulur.
  - sentence_model ve command_model artık TEK BİR SimpleWord2Vec nesnesidir.
  - Cümle kelimeleri VE komut token'ları (mkdir, <dir>, <end> …) aynı
    vocab'a ve aynı W_in / W_out matrislerine yazılır.
  - Bu sayede <dir> gibi özel token'ların tek, tutarlı vektörü olur.
  - --db-commands ve --ctokenizer argümanları kaldırıldı (artık gereksiz).
  - Yeni --no-command-tokens flag'i: komut token'larını eğitime katmamak
    isteyenler için (genellikle kaldırılmamalıdır).
"""

import random
import math
import sqlite3
import sys
import re
import argparse
from helpers import helpers as hp, TokenizerMode, EMB_SIZE, EMB_RANGE, exp, log, tanh, COMMAND_TOKENS, KNOWN_COMMANDS
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
# MODEL  (Unified — tek vocab)
# =========================
class SimpleWord2Vec:
    """
    Hem cümle kelimelerini hem de komut token'larını aynı embedding
    uzayında öğrenen Word2Vec modeli.

    DÜZELTME: is_command_model parametresi artık yok — tek model
    her iki veri kaynağını da öğrenir.
    """
    def __init__(
        self,
        activation_type="sigmoid",
        tokenizer_mode=TokenizerMode.WORD,
        subword_n=3,
        bpe_tokenizer=None
    ):
        self.vocab = {}
        self.W_in = {}
        self.W_out = {}
        self.vocab_list = []

        self.tokenizer_mode = tokenizer_mode
        self.subword_n = subword_n
        self.bpe_tokenizer = bpe_tokenizer
        self.act_min, self.act_max, self.activation = hp.get_activation_info(activation_type)
        self.activation_type = activation_type

    def _ensure_token(self, w):
        """Token yoksa vocab'a ekle."""
        if w not in self.vocab:
            self.vocab[w] = 1
            self.W_in[w] = random_vector()
            self.W_out[w] = random_vector()
            self.vocab_list.append(w)

    def add_sentence(self, sentence, is_command=False):
        """
        Cümleyi tokenize edip vocab'a ekler.
        is_command=True iken komut parametreleri normalize edilir.
        Her iki çağrı da aynı vocab'a yazar.
        """
        tokens = hp.tokenize(
            sentence,
            is_command=is_command,
            mode=self.tokenizer_mode,
            subword_n=self.subword_n,
            bpe_tokenizer=self.bpe_tokenizer
        )
        for w in tokens:
            self._ensure_token(w)
        return tokens

    def ensure_special_tokens(self):
        """
        COMMAND_TOKENS ve KNOWN_COMMANDS'ın vocab'da olmasını garantiler.
        Eğitim verisi az olsa bile bu token'lar mutlaka embedding alır.
        """
        for tok in COMMAND_TOKENS:
            self._ensure_token(tok)
        for cmd in KNOWN_COMMANDS:
            self._ensure_token(cmd)

    def train_sentence(self, sent):
        W_in = self.W_in
        W_out = self.W_out
        vocab_list = self.vocab_list
        activation = self.activation

        for _ in range(EPOCHS):
            L = len(sent)
            for i in range(L):
                w = sent[i]
                if w not in W_in:
                    continue
                vin = W_in[w]

                start = max(0, i - WINDOW)
                end = min(L, i + WINDOW + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    # POSITIVE
                    w_out = sent[j]
                    if w_out not in W_out:
                        continue
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
                        if neg not in W_out:
                            continue
                        vneg = W_out[neg]

                        score = dot(vin, vneg)
                        activated_score = activation(score)
                        error = -activated_score
                        grad = LR * error

                        for k in EMB_RANGE:
                            tmp = vin[k]
                            vin[k] += grad * vneg[k]
                            vneg[k] += grad * tmp

    def sentence_embedding(self, sentence, is_command=False):
        tokens = hp.tokenize(
            sentence,
            is_command=is_command,
            mode=self.tokenizer_mode,
            subword_n=self.subword_n,
            bpe_tokenizer=self.bpe_tokenizer
        )

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
        """Tüm kelime vektörlerini aktivasyon aralığına göre normalize eder."""
        print(f"[INFO] Vektörler normalize ediliyor... [{self.act_min}, {self.act_max}]")
        for word in self.W_in:
            self.W_in[word] = hp.normalize_vector(self.W_in[word], self.act_min, self.act_max)
            self.W_out[word] = hp.normalize_vector(self.W_out[word], self.act_min, self.act_max)


# =========================
# KAYIT  —  Unified DB
# =========================
def save_sqlite(model, db_name="embeddings.db"):
    """
    Tek bir DB'ye yazar. Her token için 'token_type' sütunu eklendi:
      'command' → COMMAND_TOKENS veya KNOWN_COMMANDS
      'word'    → sıradan cümle kelimesi
    Bu sütun C++ tarafında wordEmbeddings / commandEmbeddings ayrımı için
    kullanılır (load_embeddings_sqlite fonksiyonu güncellendi).
    """
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        word TEXT PRIMARY KEY,
        vector TEXT,
        activation_type TEXT,
        act_min REAL,
        act_max REAL,
        token_type TEXT DEFAULT 'word'
    )
    """)

    # Mevcut tabloda token_type sütunu yoksa ekle (geriye uyumluluk)
    cur.execute("PRAGMA table_info(embeddings)")
    cols = [row[1] for row in cur.fetchall()]
    if "token_type" not in cols:
        cur.execute("ALTER TABLE embeddings ADD COLUMN token_type TEXT DEFAULT 'word'")

    for word, vec in model.W_in.items():
        # Token tipi belirle
        if word in COMMAND_TOKENS or word in KNOWN_COMMANDS:
            token_type = "command"
        else:
            token_type = "word"

        cur.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
            (
                word,
                ",".join(map(str, vec)),
                model.activation_type,
                model.act_min,
                model.act_max,
                token_type
            )
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
    cur.execute("SELECT word, vector, activation_type, act_min, act_max, token_type FROM embeddings LIMIT 5")
    for row in cur.fetchall():
        word, vector, act_type, act_min, act_max, token_type = row
        vec_values = vector.split(',')
        print(f"\nWord: {word}  [{token_type}]")
        print(f"  Activation: {act_type} [{act_min}, {act_max}]")
        print(f"  Vector (first 5 dims): {vec_values[:5]}")

    total = cur.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]
    word_count = cur.execute("SELECT COUNT(*) FROM embeddings WHERE token_type='word'").fetchone()[0]
    cmd_count  = cur.execute("SELECT COUNT(*) FROM embeddings WHERE token_type='command'").fetchone()[0]
    print(f"\nTotal tokens : {total}")
    print(f"  word       : {word_count}")
    print(f"  command    : {cmd_count}")

    conn.close()


# =========================
# DOSYA OKUMA
# =========================
def load_file(filename, end_suffix=""):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [l.strip() + end_suffix for l in f if l.strip()]
        return lines
    except FileNotFoundError:
        print(f"[ERROR] '{filename}' dosyası bulunamadı!")
        return None


# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Embedding Oluşturucu — tek DB, cümle+komut token'ları aynı uzayda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s --activation sigmoid --lr 0.1 --epochs 100
  %(prog)s -a tanh -l 0.2 -e 200 --tokenizer subword --subword-n 4
  %(prog)s --tokenizer bpe --bpe-vocab 2000
  %(prog)s --help

Mevcut Aktivasyon Fonksiyonları:
  sigmoid, tanh, relu, leaky_relu, elu, softplus, linear

Not:
  Eski sistemde 2 ayrı DB vardı (embeddings.db + embeddingsForCommands.db).
  Yeni sistemde tek embeddings.db — token_type sütunu 'word'/'command' ayrımını saklar.
  commandCreator.py ve Buildv1_3_2.cpp de buna göre güncellendi.
        """
    )

    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=['sigmoid','tanh','relu','leaky_relu','elu','softplus','linear'],
                        help='Aktivasyon fonksiyonu (default: sigmoid)')
    parser.add_argument('-l', '--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Epoch sayısı (default: 100)')
    parser.add_argument('-t', '--tokenizer', type=str, default='word',
                        choices=['word', 'subword', 'bpe'],
                        help='Tokenizer modu (default: word)')
    parser.add_argument('-n', '--subword-n', type=int, default=3,
                        help='Subword n-gram değeri (default: 3)')
    parser.add_argument('--bpe-vocab', type=int, default=2000,
                        help='BPE vocab boyutu (default: 2000)')
    parser.add_argument('--bpe-model', type=str, default='bpe_tokenizer.json',
                        help='BPE model dosya adı (default: bpe_tokenizer.json)')
    parser.add_argument('-s', '--sentences', type=str, default='sentences.txt',
                        help='Cümleler dosyası (default: sentences.txt)')
    parser.add_argument('-c', '--commands', type=str, default='commandVecs.txt',
                        help='Komutlar dosyası (default: commandVecs.txt)')
    parser.add_argument('-d', '--db', type=str, default='embeddings.db',
                        help='Unified embedding DB (default: embeddings.db)')

    args = parser.parse_args()

    # Global değişkenleri güncelle
    LR = args.lr
    EPOCHS = args.epochs

    print("""
╔══════════════════════════════════════════╗
║   UNİFİED EMBEDDİNG OLUŞTURUCU (v2)     ║
║   Tek DB — cümle + komut token'ları      ║
║   + AKTİVASYON FONKSİYONLU SÜRÜM  ✔    ║
║   + PARAMETRE TİPLEME SİSTEMİ     ✔    ║
║   + BPE TOKENIZER DESTEĞİ         ✔    ║
╚══════════════════════════════════════════╝
""")

    print(f"[AYARLAR]")
    print(f"  Aktivasyon  : {args.activation}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Tokenizer   : {args.tokenizer}")
    if args.tokenizer == 'subword':
        print(f"  Subword N   : {args.subword_n}")
    if args.tokenizer == 'bpe':
        print(f"  BPE Vocab   : {args.bpe_vocab}")
        print(f"  BPE Model   : {args.bpe_model}")
    print(f"  Cümleler    : {args.sentences}")
    print(f"  Komutlar    : {args.commands}")
    print(f"  Unified DB  : {args.db}")
    print()

    # Dosyaları yükle
    print("[INFO] Dosyalar okunuyor...")
    sentences = load_file(args.sentences)
    commands  = load_file(args.commands)   # <end> commandCreator'da ekleniyor

    if sentences is None or commands is None:
        print("[ERROR] Dosyalar yüklenemedi!")
        exit(1)

    if not sentences or not commands:
        print("[ERROR] Dosyalar boş!")
        exit(1)

    print(f"✓ {len(sentences)} cümle")
    print(f"✓ {len(commands)} komut")
    print()

    # BPE Tokenizer
    tokenizer = None
    if args.tokenizer == TokenizerMode.BPE:
        print(f"[BPE] Tokenizer oluşturuluyor (vocab_size={args.bpe_vocab})...")
        tokenizer = bpe.ByteBPETokenizer(vocab_size=args.bpe_vocab)

        all_words = []
        for line in sentences:
            all_words.extend(line.split())
        # Komut satırlarından komut kelimelerini de eğitime ekle
        for line in commands:
            tokens_cmd = hp.normalize_command_params(line.split())
            for t in tokens_cmd:
                if t not in COMMAND_TOKENS:
                    all_words.append(t)

        vocab_size = tokenizer.train(all_words)
        print(f"[BPE] Training vocab size: {vocab_size}")
        tokenizer.save(args.bpe_model)
        print(f"[BPE] Tokenizer kaydedildi: {args.bpe_model}\n")

    # Unified model oluştur
    print("[MODEL] Unified embedding modeli oluşturuluyor...")
    model = SimpleWord2Vec(
        activation_type=args.activation,
        tokenizer_mode=args.tokenizer,
        subword_n=args.subword_n,
        bpe_tokenizer=tokenizer
    )

    # Özel token'ları önceden vocab'a ekle (az veri olsa bile vektör alırlar)
    model.ensure_special_tokens()

    # ——— Cümle eğitimi ———
    print("\n[TRAIN] Cümle eğitimi başlıyor...\n")
    for i, line in enumerate(sentences, 1):
        tokens = model.add_sentence(line, is_command=False)

        if i <= 3:
            print(f"[DEBUG] Cümle {i}: '{line}'")
            print(f"[DEBUG] Token'lar: {tokens[:10]}")

        model.train_sentence(tokens)

        bar = "█" * int(i / len(sentences) * 30)
        bar = bar.ljust(30, "-")
        mode_label = "BPE" if args.tokenizer == "bpe" else args.tokenizer.upper()
        vec = model.sentence_embedding(line)
        print(f"\r[{bar}] {i}/{len(sentences)} [{mode_label}] | Vec[0]={vec[0]:.4f}", end="")

    # ——— Komut eğitimi ———
    print("\n\n[TRAIN] Komut eğitimi başlıyor (parametre tipleme aktif)...\n")
    for i, line in enumerate(commands, 1):
        # is_command=True → normalize_command_params çalışır
        tokens = model.add_sentence(line, is_command=True)
        model.train_sentence(tokens)

        bar = "█" * int(i / len(commands) * 30)
        bar = bar.ljust(30, "-")
        vec = model.sentence_embedding(line, is_command=True)
        print(f"\r[{bar}] {i}/{len(commands)} [CMD] | Vec[0]={vec[0]:.4f}", end="")

    print()

    # Normalize et
    model.normalize_all_embeddings()

    # Unified DB'ye kaydet
    save_sqlite(model, args.db)

    print(f"\n\n✓ Eğitim tamamlandı → {args.db}")
    print(f"✓ Toplam vocab boyutu: {len(model.vocab)}")

    print("\n[CHECK] Unified Embeddings DB")
    check_database(args.db)

    print("\n" + "=" * 50)
    print("ÖZET:")
    print(f"  Aktivasyon    : {model.activation_type}")
    print(f"  Çıktı aralığı : [{model.act_min}, {model.act_max}]")
    print(f"  Tokenizer     : {args.tokenizer}")
    if args.tokenizer == 'bpe':
        print(f"  BPE vocab     : {args.bpe_vocab}")
        print(f"  BPE model     : {args.bpe_model}")
    print(f"  Cümleler      : {len(sentences)}")
    print(f"  Komutlar      : {len(commands)}")
    print(f"  Toplam token  : {len(model.vocab)}")
    print(f"  Unified DB    : {args.db}")
    print("\n[ÖNEMLİ] Inference sırasında:")
    print("  1. Kullanıcı girdisini sentence_embedding ile vektörleştir")
    print("  2. Ağ çıktısını commandEmbeddings üzerinden eşleştir")
    print("  3. Aynı DB kullanıldığından <dir> vb. tek tutarlı vektöre sahip")
    print("=" * 50)