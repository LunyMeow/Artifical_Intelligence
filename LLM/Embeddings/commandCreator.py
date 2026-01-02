import sqlite3
import csv

# =========================
# AYARLAR
# =========================
EMB_SIZE = 50

# =========================
# EMBEDDING YÜKLEYICI
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
        return {}
    
    embeddings = {}
    
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
        cur.execute(f"SELECT word, vector FROM {table_name}")
        for row in cur.fetchall():
            word = row[0]
            vec_str = row[1]
            vec = [float(x) for x in vec_str.split(",")]
            if len(vec) == EMB_SIZE:
                embeddings[word] = vec
    except sqlite3.OperationalError as e:
        print(f"[ERROR] SQL hatası: {e}")
        
    conn.close()
    return embeddings

# =========================
# SENTENCE EMBEDDING
# =========================
def sentence_embedding(sentence, embeddings):
    tokens = sentence.lower().split()
    result = [0.0] * EMB_SIZE
    count = 0
    
    for word in tokens:
        if word in embeddings:
            for i in range(EMB_SIZE):
                result[i] += embeddings[word][i]
            count += 1
    
    if count > 0:
        for i in range(EMB_SIZE):
            result[i] /= count
    
    return result

# =========================
# CSV OLUŞTURUCU
# =========================
def create_command_dataset(
    sentences_file="sentences.txt",
    commands_file="commandVecs.txt",
    sentence_db="embeddings.db",
    command_db="embeddingsForCommands.db",
    output_file="command_data.csv"
):
    print("Sentence embeddings yükleniyor...")
    sentence_embeddings = load_embeddings_sqlite(sentence_db)

    print("Command embeddings yükleniyor...")
    command_embeddings = load_embeddings_sqlite(command_db)

    if not sentence_embeddings or not command_embeddings:
        print("[ERROR] Embeddings yüklenemedi!")
        return

    print(f"✓ Sentence kelime sayısı: {len(sentence_embeddings)}")
    print(f"✓ Command kelime sayısı: {len(command_embeddings)}\n")

    # Dosyaları oku
    try:
        with open(sentences_file, "r", encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]
        with open(commands_file, "r", encoding="utf-8") as f:
            commands = [l.strip() for l in f if l.strip()]
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if len(sentences) != len(commands):
        print("[ERROR] Cümle ve komut sayıları eşleşmiyor!")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        header = [f"input{i+1}" for i in range(EMB_SIZE)]
        header += [f"target{i+1}" for i in range(EMB_SIZE)]
        writer.writerow(header)

        processed = 0
        skipped = 0

        for i, (sentence, command) in enumerate(zip(sentences, commands), 1):

            input_vec = sentence_embedding(sentence, sentence_embeddings)
            target_vec = sentence_embedding(command, command_embeddings)

            if sum(input_vec) == 0:
                print(f"[ATLANDI] Sentence embedding yok: {sentence}")
                skipped += 1
                continue

            if sum(target_vec) == 0:
                print(f"[ATLANDI] Command embedding yok: {command}")
                skipped += 1
                continue

            writer.writerow(input_vec + target_vec)
            processed += 1

            bar = "█" * int(i / len(sentences) * 30)
            bar = bar.ljust(30, "-")
            print(f"\r[{bar}] {i}/{len(sentences)} | OK:{processed} SKIP:{skipped}", end="")

    print("\n\n✓ Dataset oluşturuldu")
    print(f"  İşlenen: {processed}")
    print(f"  Atlanan: {skipped}")
    print(f"  Dosya: {output_file}")


# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════╗
║   KOMUT TAHMİN DATASETİ OLUŞTURUCU   ║
╚═══════════════════════════════════════╝

DOSYA FORMATI:

sentences.txt (her satır bir cümle):
  terminal aç
  dosya listele
  klasör oluştur

commandVecs.txt (her satır bir komut, aynı sırada):
  bash
  ls
  mkdir

ADIMLAR:
1. createEmbeddings.py çalıştırın
2. Bu scripti çalıştırın
""")
    
    sentences_file = input("Cümleler dosyası [sentences.txt]: ").strip() or "sentences.txt"
    commands_file = input("Komutlar dosyası [commandVecs.txt]: ").strip() or "commandVecs.txt"
    output_file = input("Çıktı dosyası [command_data.csv]: ").strip() or "command_data.csv"
    
    create_command_dataset(
        sentences_file,
        commands_file,
        "embeddings.db",
        "embeddingsForCommands.db",
        output_file
    )
"""


"""