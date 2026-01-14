import sqlite3

def check_tokenization_type(db_name="embeddings.db"):
    """
    Veritabanındaki token'ların tipini kontrol eder
    """
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    # İlk 20 token'ı al
    cur.execute("SELECT word FROM embeddings LIMIT 20")
    tokens = [row[0] for row in cur.fetchall()]
    
    print(f"📊 Veritabanı: {db_name}")
    print(f"📝 Toplam token sayısı: {cur.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]}")
    print(f"\n🔍 İlk 20 token:")
    print("-" * 50)
    
    word_level = 0
    subword_level = 0
    
    for token in tokens:
        # Token uzunluğu ve içeriğine göre analiz
        if len(token) <= 3 and not token.startswith("<"):
            subword_level += 1
            marker = "🔹 [SUBWORD]"
        elif " " in token or len(token) > 10:
            word_level += 1
            marker = "🔸 [WORD]"
        else:
            marker = "❓ [UNKNOWN]"
        
        print(f"{marker:15} '{token}' (len={len(token)})")
    
    print("-" * 50)
    print(f"\n📈 Analiz:")
    print(f"  Subword benzeri: {subword_level}")
    print(f"  Word benzeri: {word_level}")
    
    if subword_level > word_level:
        print("\n✅ BPE/Subword tokenizer kullanılmış gibi görünüyor!")
    else:
        print("\n⚠️  WORD tokenizer kullanılmış gibi görünüyor!")
        print("   BPE modunda çalıştırmak için:")
        print("   python createEmbeddings.py --tokenizer bpe --bpe-vocab 2000")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    db_name = sys.argv[1] if len(sys.argv) > 1 else "embeddings.db"
    
    try:
        check_tokenization_type(db_name)
    except Exception as e:
        print(f"❌ Hata: {e}")
        print(f"Veritabanı '{db_name}' bulunamadı veya okunamadı!")