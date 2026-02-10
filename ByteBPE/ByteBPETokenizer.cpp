#include "ByteBPETokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

ByteBPETokenizer::ByteBPETokenizer(int vocab_size) : vocab_size_(vocab_size)
{
    for (int i = 0; i < 256; ++i)
    {
        ByteSeq single_byte = {static_cast<uint8_t>(i)};
        token_to_id_[single_byte] = i;
        id_to_token_[i] = single_byte;
    }
}

ByteBPETokenizer::ByteSeq ByteBPETokenizer::byte_encode(const std::string &text)
{
    ByteSeq result;
    for (char c : text)
    {
        result.push_back(static_cast<uint8_t>(c));
    }
    return result;
}

std::map<ByteBPETokenizer::BytePair, int> ByteBPETokenizer::get_pair_counts(
    const std::vector<std::vector<ByteSeq>> &sequences)
{
    std::map<BytePair, int> pairs;
    for (const auto &seq : sequences)
    {
        for (size_t i = 0; i + 1 < seq.size(); ++i)
        {
            pairs[{seq[i], seq[i + 1]}]++;
        }
    }
    return pairs;
}

std::pair<std::vector<std::vector<ByteBPETokenizer::ByteSeq>>, ByteBPETokenizer::ByteSeq>
ByteBPETokenizer::merge_pair(
    const BytePair &pair,
    const std::vector<std::vector<ByteSeq>> &sequences)
{
    const auto &a = pair.first;
    const auto &b = pair.second;
    ByteSeq ab = a;
    ab.insert(ab.end(), b.begin(), b.end());

    std::vector<std::vector<ByteSeq>> new_sequences;
    for (const auto &seq : sequences)
    {
        std::vector<ByteSeq> new_seq;
        size_t i = 0;
        size_t L = seq.size();
        while (i < L)
        {
            if (i + 1 < L && seq[i] == a && seq[i + 1] == b)
            {
                new_seq.push_back(ab);
                i += 2;
            }
            else
            {
                new_seq.push_back(seq[i]);
                i += 1;
            }
        }
        new_sequences.push_back(new_seq);
    }
    return {new_sequences, ab};
}

int ByteBPETokenizer::train(const std::vector<std::string> &texts)
{
    if (debugLog)
        std::cout << "[BPE Train] Egitim basliyor. Metin sayisi: " << texts.size() << std::endl;

    std::vector<std::vector<ByteSeq>> sequences;
    for (const auto &text : texts)
    {
        ByteSeq encoded = byte_encode(text);
        std::vector<ByteSeq> seq;
        for (uint8_t b : encoded)
        {
            seq.push_back({b});
        }
        sequences.push_back(seq);
    }

    int next_id = token_to_id_.size();
    while (token_to_id_.size() < static_cast<size_t>(vocab_size_))
    {
        auto pair_counts = get_pair_counts(sequences);
        if (pair_counts.empty())
            break;

        BytePair best_pair;
        int max_count = 0;
        for (const auto &[pair, count] : pair_counts)
        {
            if (count > max_count)
            {
                max_count = count;
                best_pair = pair;
            }
        }

        if (max_count < 1)
            break;

        auto [new_sequences, new_token] = merge_pair(best_pair, sequences);
        sequences = new_sequences;

        if (token_to_id_.find(new_token) != token_to_id_.end())
            continue;

        if (debugLog)
        {
            std::cout << "[BPE Train] Yeni merge: " << bytes_to_hex(best_pair.first)
                      << " + " << bytes_to_hex(best_pair.second)
                      << " (Frekans: " << max_count << ")" << std::endl;
        }

        token_to_id_[new_token] = next_id;
        id_to_token_[next_id] = new_token;
        merges_.push_back(best_pair);
        next_id++;
    }

    if (debugLog)
        std::cout << "[BPE Train] Tamamlandi. Vocab Boyutu: " << token_to_id_.size() << std::endl;
    return token_to_id_.size();
}

std::vector<int> ByteBPETokenizer::encode(const std::string &text)
{
    if (debugLog)
        std::cout << "[BPE Encode] Girdi: " << text << std::endl;

    // 1. Byte-level başla
    std::vector<ByteSeq> seq;
    for (uint8_t b : byte_encode(text))
    {
        seq.push_back({b});
    }

    if (debugLog)
    {
        std::cout << "[BPE Encode] Baslangic byte tokenler: \n";
        for (const auto &token : seq)
        {
            std::cout << "token(hex)= " << bytes_to_hex(token) << " \n";
        }
        std::cout << std::endl;
    }

    // 2. MERGE LOOP - merges_ listesindeki sıraya göre

    int merge_count = 0;

    for (const auto &merge_pair : merges_)
    {

        const auto &a = merge_pair.first;
        const auto &b = merge_pair.second;

        // Merge edilen yeni token
        ByteSeq merged = a;
        merged.insert(merged.end(), b.begin(), b.end());

        // Sequence'de bu pair'i bul ve merge et
        for (size_t i = 0; i + 1 < seq.size();)
        {
            if (seq[i] == a && seq[i + 1] == b)
            {
                if (debugLog)
                {
                    std::cout << "[BPE Encode] Merge #" << ++merge_count << ": "
                              << bytes_to_hex(a) << " + " << bytes_to_hex(b)
                              << " -> " << bytes_to_hex(merged) << std::endl;
                }
                seq[i] = merged;
                seq.erase(seq.begin() + i + 1);
                // Aynı pozisyondan devam et
            }
            else
            {
                i++;
            }
        }
    }

    // 3. ID mapping
    std::vector<int> ids;
    if (debugLog)
        std::cout << "[BPE Encode] Final tokenler: ";

    for (const auto &token : seq)
    {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end())
        {
            ids.push_back(it->second);
            if (debugLog)
            {
                std::cout << bytes_to_hex(token) << "(" << it->second << ") ";
            }
        }
        else
        {
            if (debugLog)
            {
                std::cout << "[HATA:Token bulunamadi:" << bytes_to_hex(token) << "] ";
            }
        }
    }

    if (debugLog)
        std::cout << std::endl;
    return ids;
}

std::string ByteBPETokenizer::decode(const std::vector<int> &ids)
{
    std::string result;
    for (int id : ids)
    {
        // Defensive: check for invalid IDs
        if (id < 0 || id > 100000)
        {
            if (debugLog)
            {
                std::cout << "[BPE Decode HATA] Gecersiz ID (out of range): " << id << std::endl;
            }
            continue;
        }
        if (id_to_token_.count(id))
        {
            const auto &bytes = id_to_token_[id];
            for (uint8_t b : bytes)
                result += static_cast<char>(b);
        }
        else if (debugLog)
        {
            std::cout << "[BPE Decode HATA] Gecersiz ID (not in map): " << id << std::endl;
        }
    }
    return result;
}

std::string ByteBPETokenizer::bytes_to_hex(const ByteSeq &bytes)
{
    //std::cout << "[BPE bytes_to_hex] bytes.size() = " << bytes.size() << std::endl;

    std::ostringstream oss;
    for (uint8_t b : bytes)
    {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

ByteBPETokenizer::ByteSeq ByteBPETokenizer::hex_to_bytes(const std::string &hex)
{
    ByteSeq bytes;
    for (size_t i = 0; i < hex.length(); i += 2)
    {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

void ByteBPETokenizer::save(const std::string &path)
{
    if (debugLog)
        std::cout << "[BPE Save] Kaydediliyor: " << path << std::endl;
    try
    {
        json j;
        json vocab_json;
        for (const auto &[token, id] : token_to_id_)
            vocab_json[bytes_to_hex(token)] = id;
        j["vocab"] = vocab_json;

        json merges_json = json::array();
        for (const auto &[a, b] : merges_)
            merges_json.push_back({bytes_to_hex(a), bytes_to_hex(b)});
        j["merges"] = merges_json;

        std::ofstream file(path);
        if (!file.is_open())
        {
            std::cerr << "[BPE Save] [Error] Dosya açılamadı";

            // throw std::runtime_error("Dosya acilamadi");
        }
        file << j.dump(2);
        if (debugLog)
            std::cout << "[BPE Save] Basarili." << std::endl;
    }
    catch (const std::exception &e)
    {
        if (debugLog)
            std::cerr << "[BPE Save HATA] " << e.what() << std::endl;
    }
}

bool ByteBPETokenizer::load(const std::string &path)
{
    std::string fixed_path = path;

    // 1. Sonda ".bin" varsa kaldır
    if (fixed_path.size() >= 4 &&
        fixed_path.substr(fixed_path.size() - 4) == ".bin")
    {
        fixed_path.erase(fixed_path.size() - 4);
    }

    // 2. ".json" yoksa ekle
    if (fixed_path.size() < 5 ||
        fixed_path.substr(fixed_path.size() - 5) != ".json")
    {
        fixed_path += ".json";
    }

    if (debugLog)
        std::cout << "[BPE Load] Yukleniyor: " << fixed_path << std::endl;

    std::ifstream file(fixed_path);
    if (!file.is_open())
    {
        if (debugLog)
            std::cerr << "[BPE Load HATA] Dosya bulunamadi: "
                      << fixed_path << std::endl;
        return false;
    }

    try
    {
        json j;
        file >> j;

        // ✅ CLEAR BEFORE LOAD
        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();

        // ✅ 1. JSON FORMAT KONTROLÜ
        if (!j.contains("vocab") || !j.contains("merges"))
        {
            if (debugLog)
                std::cerr << "[BPE Load HATA] JSON format hatali (vocab/merges yok)"
                          << std::endl;
            return false;
        }

        // ✅ 2. ÖNCE BYTE VOCAB'I EKLE (0-255)
        // Bu kritik - base tokenlar her zaman mevcut olmalı
        for (int i = 0; i < 256; ++i)
        {
            ByteSeq single_byte = {static_cast<uint8_t>(i)};
            token_to_id_[single_byte] = i;
            id_to_token_[i] = single_byte;
        }

        // ✅ 3. JSON'DAKI TÜM TOKENLARI EKLE (0-255 DAHIL TEKRAR YAZABILIYOR OLMASI LAZIM)
        // JSON'daki vocab'da tüm tokenlar vardır, ID'ye bakarak tek tek ekle
        for (auto &[hex_key, id_json] : j["vocab"].items())
        {
            int id = id_json.get<int>();
            ByteSeq token = hex_to_bytes(hex_key);

            // Tüm tokenları ekle (base bytes 0-255 ve merged tokens 256+)
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }

        // ✅ 4. MERGES LİSTESİNİ YÜKLE - ÇOK ÖNEMLİ!
        for (const auto &merge_pair : j["merges"])
        {
            ByteSeq first = hex_to_bytes(merge_pair[0]);
            ByteSeq second = hex_to_bytes(merge_pair[1]);
            merges_.push_back({first, second});
        }

        if (debugLog)
        {
            std::cout << "[BPE Load] Basarili. Vocab: "
                      << token_to_id_.size()
                      << ", Merges: " << merges_.size() << std::endl;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        if (debugLog)
        {
            std::cerr << "[BPE Load HATA] JSON ayrıştırma hatası: "
                      << e.what() << std::endl;
        }
        return false;
    }
}