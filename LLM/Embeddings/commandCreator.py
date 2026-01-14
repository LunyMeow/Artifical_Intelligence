# CSV oluşturucu - createEmbeddings.py ile tam uyumlu (BPE desteği eklenmiş)
import sqlite3
import csv
import re
import argparse
from helpers import helpers as hp
import bytebpe as bpe
import random
import time


# =========================
# AYARLAR
# =========================
EMB_SIZE = 50
EMB_RANGE = range(EMB_SIZE)

# =========================
# TOKENIZER MODE
# =========================
class TokenizerMode:
    WORD = "word"
    SUBWORD = "subword"
    BPE = "bpe"

# =========================
# NORMALIZATION
# =========================
def normalize_vector(vec, min_val, max_val):
    """
    Vektörü aktivasyon fonksiyonunun çıktı aralığına göre normalize eder
    """
    vec_min = min(vec)
    vec_max = max(vec)
    
    if vec_max == vec_min:
        normalized = [(min_val + max_val) / 2.0] * len(vec)
    else:
        normalized = [(v - vec_min) / (vec_max - vec_min) for v in vec]
        
        if min_val != float('-inf') and max_val != float('inf'):
            normalized = [min_val + v * (max_val - min_val) for v in normalized]
        elif max_val != float('inf'):
            normalized = [min_val + v * max_val for v in normalized]
        elif min_val != float('-inf'):
            normalized = [min_val + v * abs(min_val) for v in normalized]
    
    return normalized

# =========================
# TOKENIZER
# =========================
def tokenize(
    sentence,
    is_command=False,
    mode=TokenizerMode.WORD,
    subword_n=3,
    bpe_tokenizer=None
):
    """
    createEmbeddings.py'deki tokenize metoduyla aynı
    """
    sentence = sentence.lower()

    # =========================
    # BPE TOKENIZER
    # =========================
    if mode == TokenizerMode.BPE:
        if not bpe_tokenizer:
            raise ValueError("BPE tokenizer verilmedi ancak BPE modu seçildi")
        
        tokens = []
        words = sentence.split()
        
        # Komut modeli ise normalize et
        if is_command:
            words = hp.normalize_command_params(words)
        
        for w in words:
            # Özel token kontrolü
            if w.startswith("<") and w.endswith(">"):
                tokens.append(w)
                continue
            
            # BPE encode
            try:
                ids = bpe_tokenizer.encode(w)
                for token_id in ids:
                    # ID'den token string'e dönüştür
                    if token_id in bpe_tokenizer.id_to_token:
                        token_bytes = bpe_tokenizer.id_to_token[token_id]
                        token_str = token_bytes.decode("utf-8", errors="ignore")
                        tokens.append(token_str)
            except Exception as e:
                print(f"[WARNING] BPE encode hatası '{w}': {e}")
                tokens.append(w)
        
        return tokens

    # =========================
    # WORD TOKENIZER
    # =========================
    if mode == TokenizerMode.WORD:
        tokens = sentence.split()

        if is_command:
            tokens = hp.normalize_command_params(tokens)

        return tokens

    # =========================
    # SUBWORD TOKENIZER (char n-gram)
    # =========================
    if mode == TokenizerMode.SUBWORD:
        tokens = []

        # ÖNCE word-level ayır
        words = sentence.split()

        # Komut modeli ise normalize et
        if is_command:
            words = hp.normalize_command_params(words)

        for w in words:
            # 1️⃣ ÖZEL TOKEN → aynen bırak
            if w.startswith("<") and w.endswith(">"):
                tokens.append(w)
                continue

            # 2️⃣ NORMAL KELİME → subword
            L = len(w)
            if L < subword_n:
                tokens.append(w)  # ✅ Kısa kelimeler olduğu gibi ekleniyor
                continue

            for i in range(L - subword_n + 1):
                tokens.append(w[i:i + subword_n])

        return tokens

    return []

# =========================
# EMBEDDING YÜKLEYİCİ (Activation info ile)
# =========================
def load_embeddings_sqlite(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Önce tabloları kontrol et
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    
    if not tables:
        print("[ERROR] Veritabanında hiç tablo yok!")
        print("\n[ÇÖZÜM] Önce embeddings oluşturun:")
        print("  python3 createEmbeddings.py")
        conn.close()
        return {}, None, None, None
    
    embeddings = {}
    activation_type = None
    act_min = None
    act_max = None
    
    # Tablo adını bul
    table_name = None
    for table in tables:
        if 'embedding' in table[0].lower():
            table_name = table[0]
            break
    
    if not table_name:
        table_name = tables[0][0]
    
    print(f"[INFO] '{table_name}' tablosu kullanılıyor...")
    
    try:
        cur.execute(f"SELECT word, vector, activation_type, act_min, act_max FROM {table_name}")
        rows = cur.fetchall()
        
        for row in rows:
            word = row[0]
            vec_str = row[1]
            vec = [float(x) for x in vec_str.split(",")]
            
            if len(vec) == EMB_SIZE:
                embeddings[word] = vec
            
            # İlk satırdan aktivasyon bilgisini al
            if activation_type is None:
                activation_type = row[2]
                act_min = row[3]
                act_max = row[4]
                
    except sqlite3.OperationalError as e:
        print(f"[ERROR] SQL hatası: {e}")
    
    conn.close()
    return embeddings, activation_type, act_min, act_max

# =========================
# SENTENCE EMBEDDING (createEmbeddings.py ile aynı)
# =========================
def sentence_embedding(
    sentence,
    embeddings,
    act_min,
    act_max,
    is_command=False,
    tokenizer_mode=TokenizerMode.WORD,
    subword_n=3,
    bpe_tokenizer=None
):
    """
    createEmbeddings.py'deki sentence_embedding metoduyla aynı mantık
    
    Args:
        sentence: Tokenize edilecek cümle
        embeddings: Kelime vektörleri dictionary
        act_min: Aktivasyon fonksiyonu minimum değeri
        act_max: Aktivasyon fonksiyonu maksimum değeri
        is_command: True ise parametre normalizasyonu uygula
        tokenizer_mode: TokenizerMode.WORD, SUBWORD veya BPE
        subword_n: Subword n-gram değeri
        bpe_tokenizer: BPE tokenizer objesi (sadece BPE modunda)
    """
    tokens = tokenize(
        sentence,
        is_command=is_command,
        mode=tokenizer_mode,
        subword_n=subword_n,
        bpe_tokenizer=bpe_tokenizer
    )
    
    vec = [0.0] * EMB_SIZE
    cnt = 0
    
    for w in tokens:
        if w in embeddings:
            vw = embeddings[w]
            for i in EMB_RANGE:
                vec[i] += vw[i]
            cnt += 1
    
    if cnt:
        inv = 1.0 / cnt
        for i in EMB_RANGE:
            vec[i] *= inv
    
    # createEmbeddings.py gibi normalize et
    vec = normalize_vector(vec, act_min, act_max)
    return vec

# =========================
# CSV OLUŞTURUCU
# =========================
def create_command_dataset(
    sentences_file="sentences.txt",
    commands_file="commandVecs.txt",
    sentence_db="embeddings.db",
    command_db="embeddingsForCommands.db",
    output_file="command_data.csv",
    sentence_tokenizer_mode=TokenizerMode.WORD,
    command_tokenizer_mode=TokenizerMode.WORD,
    subword_n=3,
    bpe_model_path=None,
    shuffle_data=False
):
    print("\n[INFO] Sentence embeddings yükleniyor...")
    sentence_embeddings, sent_act, sent_min, sent_max = load_embeddings_sqlite(sentence_db)

    print("[INFO] Command embeddings yükleniyor...")
    command_embeddings, cmd_act, cmd_min, cmd_max = load_embeddings_sqlite(command_db)

    if not sentence_embeddings or not command_embeddings:
        print("[ERROR] Embeddings yüklenemedi!")
        return

    print(f"✓ Sentence kelime sayısı: {len(sentence_embeddings)}")
    print(f"✓ Command token sayısı: {len(command_embeddings)}")
    print(f"✓ Sentence aktivasyon: {sent_act} [{sent_min}, {sent_max}]")
    print(f"✓ Command aktivasyon: {cmd_act} [{cmd_min}, {cmd_max}]")
    print(f"✓ Sentence tokenizer: {sentence_tokenizer_mode}")
    print(f"✓ Command tokenizer: {command_tokenizer_mode}")
    
    # BPE tokenizer yükle (gerekiyorsa)
    sentence_bpe_tokenizer = None
    command_bpe_tokenizer = None
    
    if sentence_tokenizer_mode == TokenizerMode.BPE:
        if not bpe_model_path:
            print("[ERROR] BPE modu seçildi ama --bpe-model belirtilmedi!")
            return
        
        print(f"\n[BPE] Sentence tokenizer yükleniyor: {bpe_model_path}")
        sentence_bpe_tokenizer = bpe.ByteBPETokenizer()
        sentence_bpe_tokenizer.load(bpe_model_path)
        print(f"✓ BPE vocab size: {len(sentence_bpe_tokenizer.token_to_id)}")
    
    if command_tokenizer_mode == TokenizerMode.BPE:
        if not bpe_model_path:
            print("[ERROR] Command BPE modu seçildi ama --bpe-model belirtilmedi!")
            return
        
        print(f"[BPE] Command tokenizer yükleniyor: {bpe_model_path}")
        command_bpe_tokenizer = bpe.ByteBPETokenizer()
        command_bpe_tokenizer.load(bpe_model_path)
    
    # Placeholder token'ları kontrol et
    placeholders = ["<DIR>", "<FILE>", "<PATH>", "<NUMBER>", "<IP>", "<URL>", "<VAR>", "<end>"]
    found_placeholders = [p for p in placeholders if p in command_embeddings]
    if found_placeholders:
        print(f"✓ Özel tokenler bulundu: {', '.join(found_placeholders)}\n")
    else:
        print("[UYARI] Hiç özel token bulunamadı. createEmbeddings.py çalıştırıldı mı?\n")

    # Dosyaları oku
    try:
        with open(sentences_file, "r", encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]
        
        with open(commands_file, "r", encoding="utf-8") as f:
            # <end> ekliyoruz çünkü createEmbeddings.py de ekliyor
            commands = [line.strip() + "" for line in f if line.strip()]

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if len(sentences) != len(commands):
        print(f"[ERROR] Cümle ve komut sayıları eşleşmiyor!")
        print(f"  Cümleler: {len(sentences)}")
        print(f"  Komutlar: {len(commands)}")
        return

    print(f"[INFO] CSV oluşturuluyor: {output_file}\n")
    dataset_rows = []

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        header = [f"input{i+1}" for i in range(EMB_SIZE)]
        header += [f"target{i+1}" for i in range(EMB_SIZE)]
        writer.writerow(header)

        processed = 0
        skipped = 0

        for i, (sentence, command) in enumerate(zip(sentences, commands), 1):
            # Sentence: normal embedding (parametre normalizasyonu YOK)
            input_vec = sentence_embedding(
                sentence,
                sentence_embeddings,
                sent_min,
                sent_max,
                is_command=False,
                tokenizer_mode=sentence_tokenizer_mode,
                subword_n=subword_n,
                bpe_tokenizer=sentence_bpe_tokenizer
            )
            
            # Command: parametre normalizasyonlu embedding
            target_vec = sentence_embedding(
                command,
                command_embeddings,
                cmd_min,
                cmd_max,
                is_command=True,
                tokenizer_mode=command_tokenizer_mode,
                subword_n=subword_n,
                bpe_tokenizer=command_bpe_tokenizer
            )

            # Vektör kontrolü
            input_sum = sum(abs(x) for x in input_vec)
            target_sum = sum(abs(x) for x in target_vec)

            if input_sum == 0:
                print(f"\n[ATLANDI] Sentence embedding yok: '{sentence}'")
                skipped += 1
                continue

            if target_sum == 0:
                print(f"\n[ATLANDI] Command embedding yok: '{command}'")
                skipped += 1
                continue

            dataset_rows.append(input_vec + target_vec)
            processed += 1

            # Progress bar
            if i % 10 == 0 or i == len(sentences):
                bar = "█" * int(i / len(sentences) * 30)
                bar = bar.ljust(30, "-")
                print(f"\r[{bar}] {i}/{len(sentences)} | OK:{processed} SKIP:{skipped}", end="")

        print(f"\n[INFO] Dataset satır sayısı (shuffle öncesi): {len(dataset_rows)}")

        if shuffle_data:
            print("[INFO] Dataset karıştırılıyor...")
            random.shuffle(dataset_rows)
        
        print(f"[INFO] CSV'ye yazılıyor...")
        for row in dataset_rows:
            writer.writerow(row)

    print("\n\n✓ Dataset oluşturuldu")
    print(f"  İşlenen: {processed}")
    print(f"  Atlanan: {skipped}")
    print(f"  Dosya: {output_file}")
    print(f"  Toplam satır: {processed + 1} (header dahil)")


# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Komut Tahmin Dataseti Oluşturucu - createEmbeddings.py ile tam uyumlu (BPE destekli)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s
  %(prog)s -s sentences.txt -c commandVecs.txt -o output.csv
  %(prog)s --sentence-db myemb.db --command-db mycmd.db
  %(prog)s --sentence-tokenizer bpe --bpe-model bpe_tokenizer.json
  %(prog)s --sentence-tokenizer subword --subword-n 4

DOSYA FORMATI:

sentences.txt (her satır bir cümle):
  terminal aç
  dosya listele
  klasör oluştur

commandVecs.txt (her satır bir komut, aynı sırada):
  bash
  ls
  mkdir

NOT: 
1. Önce createEmbeddings.py çalıştırılmalı
2. Komutlardaki parametreler otomatik normalize edilir:
   mkdir test_folder  →  mkdir <DIR> <end>
   rm file.txt        →  rm <FILE> <end>
3. sentences.txt ve commandVecs.txt AYNI SATIR SAYISINDA olmalı
4. Aktivasyon fonksiyonu ve normalizasyon otomatik DB'den alınır
5. BPE modu kullanıyorsanız --bpe-model parametresini belirtin
        """
    )
    
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
        '-o', '--output',
        type=str,
        default='command_data.csv',
        help='Çıktı CSV dosyası (default: command_data.csv)'
    )
    
    parser.add_argument(
        '--sentence-db',
        type=str,
        default='embeddings.db',
        help='Cümle embeddings DB (default: embeddings.db)'
    )
    
    parser.add_argument(
        '--command-db',
        type=str,
        default='embeddingsForCommands.db',
        help='Komut embeddings DB (default: embeddingsForCommands.db)'
    )

    parser.add_argument(
        '--sentence-tokenizer',
        type=str,
        default='word',
        choices=['word', 'subword', 'bpe'],
        help='Sentence tokenizer modu (default: word)'
    )
    
    parser.add_argument(
        '--command-tokenizer',
        type=str,
        default='word',
        choices=['word', 'subword', 'bpe'],
        help='Command tokenizer modu (default: word)'
    )
    
    parser.add_argument(
        '--subword-n',
        type=int,
        default=3,
        help='Subword n-gram değeri (default: 3)'
    )
    
    parser.add_argument(
        '--bpe-model',
        type=str,
        default=None,
        help='BPE model dosya yolu (BPE modu için gerekli)'
    )
    
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Dataset satırlarını CSV yazılmadan önce karıştır'
    )

    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════╗
║   KOMUT TAHMİN DATASETİ OLUŞTURUCU   ║
║  + PARAMETRE TİPLEME DESTEĞİ ✔       ║
║  + AKTİVASYON NORMALİZASYONU ✔       ║
║  + BPE TOKENIZER DESTEĞİ ✔           ║
╚═══════════════════════════════════════╝
""")
    
    print("[AYARLAR]")
    print(f"  Cümleler: {args.sentences}")
    print(f"  Komutlar: {args.commands}")
    print(f"  Çıktı: {args.output}")
    print(f"  Sentence DB: {args.sentence_db}")
    print(f"  Command DB: {args.command_db}")
    print(f"  Sentence Tokenizer: {args.sentence_tokenizer}")
    print(f"  Command Tokenizer: {args.command_tokenizer}")
    if args.sentence_tokenizer == 'subword' or args.command_tokenizer == 'subword':
        print(f"  Subword N: {args.subword_n}")
    if args.sentence_tokenizer == 'bpe' or args.command_tokenizer == 'bpe':
        print(f"  BPE Model: {args.bpe_model}")
    print(f"  Shuffle Mode: {args.shuffle}")

    
    create_command_dataset(
        args.sentences,
        args.commands,
        args.sentence_db,
        args.command_db,
        args.output,
        args.sentence_tokenizer,
        args.command_tokenizer,
        args.subword_n,
        args.bpe_model,
        args.shuffle
    )