#!/usr/bin/env python3
"""
Train command embeddings with parameters and generate embedding databases.
This script creates:
1. Word embeddings for input sentences
2. Command embeddings for template-tagged commands
3. SQLite databases for C++ integration
"""

import sqlite3
import csv
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from helpers import helpers as hp, TokenizerMode, EMB_SIZE
from createEmbeddings import SimpleWord2Vec

def load_training_data(csv_file):
    """Load training data from CSV."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'sentence': row['sentence'],
                'command': row['command'],
                'template': row['template'],
                'output': row['output'],
                'output_with_end': row['output_with_end'],
            })
    print(f"✅ Loaded {len(data)} training samples from {csv_file}")
    return data

def create_sentence_embeddings(training_data, mode=TokenizerMode.BPE):
    """Create word embeddings from input sentences."""
    print("\n📚 Training word embeddings...")
    
    model = SimpleWord2Vec(
        activation_type="sigmoid",
        is_command_model=False,
        tokenizer_mode=mode,
        normalize_command_tokens=False
    )
    
    # Collect all sentences
    sentences = [item['sentence'] for item in training_data]
    
    # Train
    for epoch in range(50):  # Fewer epochs for speed
        for sentence in sentences:
            model.train_sentence(sentence, learning_rate=0.1)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/50 complete")
    
    embeddings = {}
    for token in model.vocab:
        if token in model.W_in:
            embeddings[token] = model.W_in[token]
    
    print(f"✅ Created {len(embeddings)} word embeddings")
    return embeddings, model

def create_command_embeddings(training_data, mode=TokenizerMode.BPE):
    """Create command embeddings from template-tagged outputs."""
    print("\n🎯 Training command embeddings...")
    
    model = SimpleWord2Vec(
        activation_type="sigmoid",
        is_command_model=True,
        tokenizer_mode=mode,
        normalize_command_tokens=True
    )
    
    # Collect all command outputs
    commands = [item['output_with_end'] for item in training_data]
    
    # Train
    for epoch in range(50):
        for command in commands:
            model.train_sentence(command, learning_rate=0.1)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/50 complete")
    
    embeddings = {}
    for token in model.vocab:
        if token in model.W_in:
            embeddings[token] = model.W_in[token]
    
    print(f"✅ Created {len(embeddings)} command embeddings")
    return embeddings, model

def save_embeddings_to_sqlite(embeddings, db_file):
    """Save embeddings to SQLite database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            token TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')
    
    # Insert embeddings
    for token, vec in embeddings.items():
        # Convert list to comma-separated string
        vec_str = ','.join(str(v) for v in vec)
        cursor.execute('INSERT OR REPLACE INTO embeddings (token, embedding) VALUES (?, ?)',
                      (token, vec_str))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Saved {len(embeddings)} embeddings to {db_file}")

def generate_training_vectors_csv(embeddings, csv_file):
    """Generate CSV-style vector file for reference."""
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        f.write("token," + ",".join(f"dim_{i}" for i in range(EMB_SIZE)) + "\n")
        for token, vec in embeddings.items():
            row = token + "," + ",".join(str(v) for v in vec)
            f.write(row + "\n")
    print(f"✅ Generated vector CSV with {len(embeddings)} entries")

def main():
    import os
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    print("=" * 70)
    print(" 🚀 ENHANCED COMMAND EMBEDDING TRAINING")
    print("=" * 70)
    
    # Load training data
    training_data = load_training_data("enhanced_command_data.csv")
    
    # Create embeddings
    word_emb, word_model = create_sentence_embeddings(training_data)
    cmd_emb, cmd_model = create_command_embeddings(training_data)
    
    # Save to databases
    print("\n💾 Saving embeddings...")
    save_embeddings_to_sqlite(word_emb, "embeddings_enhanced.db")
    save_embeddings_to_sqlite(cmd_emb, "embeddingsForCommands_enhanced.db")
    
    # Generate reference CSVs
    generate_training_vectors_csv(word_emb, "word_embeddings_enhanced.csv")
    generate_training_vectors_csv(cmd_emb, "command_embeddings_enhanced.csv")
    
    print("\n" + "=" * 70)
    print(" ✅ TRAINING COMPLETE")
    print("=" * 70)
    print("\n📊 Results:")
    print(f"   Word embeddings: {len(word_emb)} tokens")
    print(f"   Command embeddings: {len(cmd_emb)} tokens")
    print("\n📁 Output files:")
    print("   - embeddings_enhanced.db (word embeddings)")
    print("   - embeddingsForCommands_enhanced.db (command embeddings)")
    print("   - word_embeddings_enhanced.csv (reference)")
    print("   - command_embeddings_enhanced.csv (reference)")
    print("\n💡 Next: Copy databases to replace existing ones and rebuild C++")
    print("=" * 70)

if __name__ == "__main__":
    main()
