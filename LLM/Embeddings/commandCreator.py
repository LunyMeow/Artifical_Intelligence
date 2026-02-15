"""
commandCreator.py  —  Unified DB ile CSV Oluşturucu
=====================================================
DEĞİŞİKLİK ÖZETİ (eski → yeni):
  - load_embeddings_sqlite: İki DB yerine TEK DB okur.
    token_type sütununu kullanarak word / command embedding'lerini
    otomatik olarak ayırır.
  - --sentence-db ve --command-db argümanı yerine tek --db argümanı.
  - sentence_embedding ve target_embedding artık aynı büyük dict'ten
    (allEmbeddings) beslenir — <dir> vb. için tek vektör var.
  - Eski uyumluluk için --sentence-db / --command-db hâlâ kabul edilir
    ama birbirinin aynısını göstermeleri beklenir; uyarı verilir.
"""

import sqlite3
import csv
import re
import argparse
from helpers import helpers as hp, TokenizerMode, EMB_SIZE, EMB_RANGE, COMMAND_TOKENS
import bytebpe as bpe
import random


# =========================
# UNIFIED EMBEDDING YÜKLEYİCİ
# =========================
def load_unified_embeddings(db_path="embeddings.db"):
    """
    Unified DB'yi okur. token_type sütununa göre word / command
    embedding'lerini iki dict'e ayırarak döndürür.

    Returns:
        all_embeddings    : tüm token'lar (word + command)
        word_embeddings   : sadece word token'ları
        command_embeddings: sadece command token'ları
        activation_type   : str
        act_min           : float
        act_max           : float
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]

    if not tables:
        print("[ERROR] Veritabanında hiç tablo yok!")
        print("\n[ÇÖZÜM] Önce embeddings oluşturun:")
        print("  python3 createEmbeddings.py")
        conn.close()
        return {}, {}, {}, None, None, None

    # Tablo adını bul
    table_name = next((t for t in tables if 'embedding' in t.lower()), tables[0])
    print(f"[INFO] '{table_name}' tablosu kullanılıyor...")

    # token_type sütunu var mı kontrol et
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    has_token_type = "token_type" in cols

    if not has_token_type:
        print("[UYARI] 'token_type' sütunu bulunamadı. Eski format DB — "
              "yeniden createEmbeddings.py çalıştırmanız önerilir.")
        print("[INFO] Geçici olarak tüm token'lar her iki dict'e de ekleniyor.")

    all_embeddings = {}
    word_embeddings = {}
    command_embeddings = {}
    activation_type = None
    act_min = None
    act_max = None

    try:
        if has_token_type:
            cur.execute(f"SELECT word, vector, activation_type, act_min, act_max, token_type FROM {table_name}")
        else:
            cur.execute(f"SELECT word, vector, activation_type, act_min, act_max FROM {table_name}")

        rows = cur.fetchall()

        for row in rows:
            word    = row[0]
            vec_str = row[1]
            vec     = [float(x) for x in vec_str.split(",")]

            if len(vec) != EMB_SIZE:
                continue

            # Aktivasyon bilgisini ilk satırdan al
            if activation_type is None:
                activation_type = row[2]
                act_min         = row[3]
                act_max         = row[4]

            all_embeddings[word] = vec

            if has_token_type:
                token_type = row[5] if row[5] else "word"
            else:
                # Eski format: COMMAND_TOKENS içindeyse command, değilse word
                token_type = "command" if word in COMMAND_TOKENS else "word"

            if token_type == "command":
                command_embeddings[word] = vec
            else:
                word_embeddings[word] = vec

    except sqlite3.OperationalError as e:
        print(f"[ERROR] SQL hatası: {e}")

    conn.close()
    return all_embeddings, word_embeddings, command_embeddings, activation_type, act_min, act_max


# =========================
# CSV OLUŞTURUCU
# =========================
def create_command_dataset(
    sentences_file="sentences.txt",
    commands_file="commandVecs.txt",
    db_path="embeddings.db",
    output_file="command_data.csv",
    sentence_tokenizer_mode=TokenizerMode.WORD,
    command_tokenizer_mode=TokenizerMode.WORD,
    subword_n=3,
    bpe_model_path=None,
    shuffle_data=False
):
    print("\n[INFO] Unified embeddings yükleniyor...")
    all_emb, word_emb, cmd_emb, act_type, act_min, act_max = load_unified_embeddings(db_path)

    if not all_emb:
        print("[ERROR] Embeddings yüklenemedi!")
        return

    print(f"✓ Toplam token sayısı   : {len(all_emb)}")
    print(f"✓ Word token sayısı     : {len(word_emb)}")
    print(f"✓ Command token sayısı  : {len(cmd_emb)}")
    print(f"✓ Aktivasyon            : {act_type} [{act_min}, {act_max}]")
    print(f"✓ Sentence tokenizer    : {sentence_tokenizer_mode}")
    print(f"✓ Command tokenizer     : {command_tokenizer_mode}")

    # BPE tokenizer yükle (gerekiyorsa)
    sentence_bpe = None
    command_bpe  = None

    if sentence_tokenizer_mode == TokenizerMode.BPE:
        if not bpe_model_path:
            print("[ERROR] BPE modu seçildi ama --bpe-model belirtilmedi!")
            return
        print(f"\n[BPE] Sentence tokenizer yükleniyor: {bpe_model_path}")
        sentence_bpe = bpe.ByteBPETokenizer()
        sentence_bpe.load(bpe_model_path)
        print(f"✓ BPE vocab size: {len(sentence_bpe.token_to_id)}")

    if command_tokenizer_mode == TokenizerMode.BPE:
        if not bpe_model_path:
            print("[ERROR] Command BPE modu seçildi ama --bpe-model belirtilmedi!")
            return
        print(f"[BPE] Command tokenizer yükleniyor: {bpe_model_path}")
        command_bpe = bpe.ByteBPETokenizer()
        command_bpe.load(bpe_model_path)

    # Özel token'ları doğrula
    found = [p for p in COMMAND_TOKENS if p in cmd_emb]
    if found:
        print(f"✓ Özel tokenler bulundu: {', '.join(found)}\n")
    else:
        print("[UYARI] Hiç özel token bulunamadı. "
              "createEmbeddings.py çalıştırıldı mı?\n")

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
        skipped   = 0

        for i, (sentence, command) in enumerate(zip(sentences, commands), 1):

            commandTokenTemp = ""
            for commandToken in command.split():

                # Komut token'ını normalize et
                norm_tokens = hp.normalize_command_params([commandToken])
                norm_token  = norm_tokens[0] if norm_tokens else commandToken

                # Input: cümle + şimdiye kadar üretilen komut parçası
                last_sentence = sentence + (" " + commandTokenTemp if commandTokenTemp else "")

                input_vec = hp.sentence_embedding(
                    last_sentence,
                    all_emb,        # ← Unified dict — hem word hem command
                    act_min,
                    act_max,
                    is_command=False,
                    tokenizer_mode=sentence_tokenizer_mode,
                    subword_n=subword_n,
                    bpe_tokenizer=sentence_bpe
                )

                # Target: tek komut token'ı (normalize edilmiş)
                # is_command=True → normalize_command_params çalışır
                # Lookup da all_emb üzerinden — tutarlı tek vektör
                target_vec = hp.sentence_embedding(
                    norm_token,
                    all_emb,        # ← Unified dict
                    act_min,
                    act_max,
                    is_command=True,
                    tokenizer_mode=command_tokenizer_mode,
                    subword_n=subword_n,
                    bpe_tokenizer=command_bpe
                )

                input_sum  = sum(abs(x) for x in input_vec)
                target_sum = sum(abs(x) for x in target_vec)

                if input_sum == 0:
                    print(f"\n[ATLANDI] Sentence embedding yok: '{last_sentence}'")
                    skipped += 1
                    continue

                if target_sum == 0:
                    print(f"\n[ATLANDI] Command embedding yok: '{command} / {norm_token}'")
                    skipped += 1
                    continue

                dataset_rows.append(input_vec + target_vec)
                processed += 1

                commandTokenTemp = norm_token

                if norm_token == "<end>":
                    break

            # Progress bar
            if i % 10 == 0 or i == len(sentences):
                bar = "█" * int(i / len(sentences) * 30)
                bar = bar.ljust(30, "-")
                print(f"\r[{bar}] {i}/{len(sentences)} | OK:{processed} SKIP:{skipped}", end="")

        print(f"\n[INFO] Dataset satır sayısı (shuffle öncesi): {len(dataset_rows)}")

        if shuffle_data:
            print("[INFO] Dataset karıştırılıyor...")
            random.shuffle(dataset_rows)

        print("[INFO] CSV'ye yazılıyor...")
        for row in dataset_rows:
            writer.writerow(row)

    print("\n\n✓ Dataset oluşturuldu")
    print(f"  İşlenen : {processed}")
    print(f"  Atlanan  : {skipped}")
    print(f"  Dosya    : {output_file}")
    print(f"  Toplam satır: {processed + 1} (header dahil)")


# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Komut Tahmin Dataseti Oluşturucu — Unified DB (BPE destekli)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s
  %(prog)s -s sentences.txt -c commandVecs.txt -o output.csv
  %(prog)s --db embeddings.db
  %(prog)s --sentence-tokenizer bpe --bpe-model bpe_tokenizer.json
  %(prog)s --sentence-tokenizer subword --subword-n 4

DEĞİŞİKLİK: Artık tek --db parametresi var (iki DB yerine).
  Eski: --sentence-db embeddings.db --command-db embeddingsForCommands.db
  Yeni: --db embeddings.db
        """
    )

    parser.add_argument('-s', '--sentences', type=str, default='sentences.txt',
                        help='Cümleler dosyası (default: sentences.txt)')
    parser.add_argument('-c', '--commands', type=str, default='commandVecs.txt',
                        help='Komutlar dosyası (default: commandVecs.txt)')
    parser.add_argument('-o', '--output', type=str, default='command_data.csv',
                        help='Çıktı CSV dosyası (default: command_data.csv)')

    # Yeni: tek DB
    parser.add_argument('--db', type=str, default='embeddings.db',
                        help='Unified embedding DB (default: embeddings.db)')

    # Geriye uyumluluk — ikisi de varsa uyarı ver
    parser.add_argument('--sentence-db', type=str, default=None,
                        help='[ESKİ] Sadece geriye uyumluluk için; --db kullanın')
    parser.add_argument('--command-db', type=str, default=None,
                        help='[ESKİ] Sadece geriye uyumluluk için; --db kullanın')

    parser.add_argument('--sentence-tokenizer', type=str, default='word',
                        choices=['word', 'subword', 'bpe'],
                        help='Sentence tokenizer modu (default: word)')
    parser.add_argument('--command-tokenizer', type=str, default='word',
                        choices=['word', 'subword', 'bpe'],
                        help='Command tokenizer modu (default: word)')
    parser.add_argument('--subword-n', type=int, default=3,
                        help='Subword n-gram değeri (default: 3)')
    parser.add_argument('--bpe-model', type=str, default=None,
                        help='BPE model dosya yolu (BPE modu için gerekli)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Dataset satırlarını CSV yazılmadan önce karıştır')

    args = parser.parse_args()

    # Geriye uyumluluk uyarısı
    db_path = args.db
    if args.sentence_db or args.command_db:
        print("[UYARI] --sentence-db ve --command-db artık kullanılmıyor.")
        print("        Lütfen --db kullanın. Devam ediliyor...\n")
        if args.sentence_db:
            db_path = args.sentence_db  # en azından bir tanesini al

    print("""
╔══════════════════════════════════════════╗
║   KOMUT TAHMİN DATASETİ OLUŞTURUCU      ║
║   Unified DB — tek embedding uzayı       ║
║   + PARAMETRE TİPLEME DESTEĞİ   ✔       ║
║   + AKTİVASYON NORMALİZASYONU   ✔       ║
║   + BPE TOKENIZER DESTEĞİ       ✔       ║
╚══════════════════════════════════════════╝
""")

    print("[AYARLAR]")
    print(f"  Cümleler         : {args.sentences}")
    print(f"  Komutlar         : {args.commands}")
    print(f"  Çıktı            : {args.output}")
    print(f"  Unified DB       : {db_path}")
    print(f"  Sentence Tokenizer: {args.sentence_tokenizer}")
    print(f"  Command Tokenizer : {args.command_tokenizer}")
    if args.sentence_tokenizer == 'subword' or args.command_tokenizer == 'subword':
        print(f"  Subword N        : {args.subword_n}")
    if args.sentence_tokenizer == 'bpe' or args.command_tokenizer == 'bpe':
        print(f"  BPE Model        : {args.bpe_model}")
    print(f"  Shuffle          : {args.shuffle}")

    create_command_dataset(
        sentences_file=args.sentences,
        commands_file=args.commands,
        db_path=db_path,
        output_file=args.output,
        sentence_tokenizer_mode=args.sentence_tokenizer,
        command_tokenizer_mode=args.command_tokenizer,
        subword_n=args.subword_n,
        bpe_model_path=args.bpe_model,
        shuffle_data=args.shuffle
    )