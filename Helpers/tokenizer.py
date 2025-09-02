# tokenize_and_decode.py
import os
import csv
import json
from transformers import GPT2Tokenizer

# Tokenizer'ı yükle
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_directory_to_csv(directory, output_csv="tokenized_output.csv"):
    """
    Dizindeki tüm txt dosyalarını tokenize edip CSV'ye kaydeder.
    """
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "line_number", "tokens"])  # başlık

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        token_ids = tokenizer.encode(line)
                        writer.writerow([filename, i, " ".join(map(str, token_ids))])
    print(f"Tüm metinler tokenize edildi ve '{output_csv}' dosyasına kaydedildi.")

def save_json_from_csv(csv_file, json_file="tokenized_output.json"):
    """
    CSV'den tokenleri okuyup JSON'a kaydeder.
    """
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_ids = list(map(int, row["tokens"].split()))
            data.append({
                "file": row["file"],
                "line_number": int(row["line_number"]),
                "tokens": token_ids
            })
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)
    print(f"CSV verileri JSON olarak kaydedildi: '{json_file}'")

def decode_from_json(json_file="tokenized_output.json"):
    """
    JSON dosyasındaki tokenleri decode edip metin olarak ekrana yazdırır.
    """
    with open(json_file, "r", encoding="utf-8") as jf:
        data = json.load(jf)
        for entry in data:
            text = tokenizer.decode(entry["tokens"])
            print(f"{entry['file']} (line {entry['line_number']}): {text}")

if __name__ == "__main__":
    directory = input("Tokenize edilecek dizin: ").strip()
    tokenize_directory_to_csv(directory)
    save_json_from_csv("tokenized_output.csv")
    print("\n--- JSON'dan Decode ---")
    decode_from_json("tokenized_output.json")
