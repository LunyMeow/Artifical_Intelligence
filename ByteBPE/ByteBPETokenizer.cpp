#include "ByteBPETokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

ByteBPETokenizer::ByteBPETokenizer(int vocab_size) : vocab_size_(vocab_size) {
    // Initialize with single bytes (0-255)
    for (int i = 0; i < 256; ++i) {
        ByteSeq single_byte = {static_cast<uint8_t>(i)};
        token_to_id_[single_byte] = i;
        id_to_token_[i] = single_byte;
    }
}

ByteBPETokenizer::ByteSeq ByteBPETokenizer::byte_encode(const std::string& text) {
    ByteSeq result;
    for (char c : text) {
        result.push_back(static_cast<uint8_t>(c));
    }
    return result;
}

std::map<ByteBPETokenizer::BytePair, int> ByteBPETokenizer::get_pair_counts(
    const std::vector<std::vector<ByteSeq>>& sequences
) {
    std::map<BytePair, int> pairs;
    for (const auto& seq : sequences) {
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            pairs[{seq[i], seq[i + 1]}]++;
        }
    }
    return pairs;
}

std::pair<std::vector<std::vector<ByteBPETokenizer::ByteSeq>>, ByteBPETokenizer::ByteSeq>
ByteBPETokenizer::merge_pair(
    const BytePair& pair,
    const std::vector<std::vector<ByteSeq>>& sequences
) {
    const auto& a = pair.first;
    const auto& b = pair.second;
    ByteSeq ab = a;
    ab.insert(ab.end(), b.begin(), b.end());
    
    std::vector<std::vector<ByteSeq>> new_sequences;
    
    for (const auto& seq : sequences) {
        std::vector<ByteSeq> new_seq;
        size_t i = 0;
        size_t L = seq.size();
        
        while (i < L) {
            if (i + 1 < L && seq[i] == a && seq[i + 1] == b) {
                new_seq.push_back(ab);
                i += 2;
            } else {
                new_seq.push_back(seq[i]);
                i += 1;
            }
        }
        new_sequences.push_back(new_seq);
    }
    
    return {new_sequences, ab};
}

int ByteBPETokenizer::train(const std::vector<std::string>& texts) {
    std::vector<std::vector<ByteSeq>> sequences;
    
    // Convert texts to byte sequences
    for (const auto& text : texts) {
        ByteSeq encoded = byte_encode(text);
        std::vector<ByteSeq> seq;
        for (uint8_t b : encoded) {
            seq.push_back({b});
        }
        sequences.push_back(seq);
    }
    
    int next_id = token_to_id_.size();
    
    while (token_to_id_.size() < static_cast<size_t>(vocab_size_)) {
        auto pair_counts = get_pair_counts(sequences);
        
        if (pair_counts.empty()) {
            break;
        }
        
        // Find best pair (most common)
        BytePair best_pair;
        int max_count = 0;
        for (const auto& [pair, count] : pair_counts) {
            if (count > max_count) {
                max_count = count;
                best_pair = pair;
            }
        }
        
        auto [new_sequences, new_token] = merge_pair(best_pair, sequences);
        sequences = new_sequences;
        
        if (token_to_id_.find(new_token) != token_to_id_.end()) {
            continue;
        }
        
        token_to_id_[new_token] = next_id;
        id_to_token_[next_id] = new_token;
        merges_.push_back(best_pair);
        next_id++;
    }
    
    return token_to_id_.size();
}

std::vector<int> ByteBPETokenizer::encode(const std::string& text) {
    ByteSeq encoded = byte_encode(text);
    std::vector<ByteSeq> seq;
    for (uint8_t b : encoded) {
        seq.push_back({b});
    }
    
    // Apply merges
    for (const auto& [a, b] : merges_) {
        ByteSeq ab = a;
        ab.insert(ab.end(), b.begin(), b.end());
        
        std::vector<ByteSeq> new_seq;
        size_t i = 0;
        size_t L = seq.size();
        
        while (i < L) {
            if (i + 1 < L && seq[i] == a && seq[i + 1] == b) {
                new_seq.push_back(ab);
                i += 2;
            } else {
                new_seq.push_back(seq[i]);
                i += 1;
            }
        }
        seq = new_seq;
    }
    
    // Convert to IDs
    std::vector<int> ids;
    for (const auto& token : seq) {
        ids.push_back(token_to_id_[token]);
    }
    return ids;
}

std::string ByteBPETokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        const auto& bytes = id_to_token_[id];
        for (uint8_t b : bytes) {
            result += static_cast<char>(b);
        }
    }
    return result;
}

std::string ByteBPETokenizer::bytes_to_hex(const ByteSeq& bytes) {
    std::ostringstream oss;
    for (uint8_t b : bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

ByteBPETokenizer::ByteSeq ByteBPETokenizer::hex_to_bytes(const std::string& hex) {
    ByteSeq bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

void ByteBPETokenizer::save(const std::string& path) {
    json j;
    
    // Save vocab
    json vocab_json;
    for (const auto& [token, id] : token_to_id_) {
        vocab_json[bytes_to_hex(token)] = id;
    }
    j["vocab"] = vocab_json;
    
    // Save merges
    json merges_json = json::array();
    for (const auto& [a, b] : merges_) {
        merges_json.push_back({bytes_to_hex(a), bytes_to_hex(b)});
    }
    j["merges"] = merges_json;
    
    std::ofstream file(path);
    file << j.dump(2);
}

void ByteBPETokenizer::load(const std::string& path) {
    std::ifstream file(path);
    json j;
    file >> j;
    
    // Load vocab
    token_to_id_.clear();
    id_to_token_.clear();
    for (auto& [hex_key, id] : j["vocab"].items()) {
        ByteSeq token = hex_to_bytes(hex_key);
        token_to_id_[token] = id;
        id_to_token_[id] = token;
    }
    
    // Load merges
    merges_.clear();
    for (const auto& merge_pair : j["merges"]) {
        ByteSeq a = hex_to_bytes(merge_pair[0]);
        ByteSeq b = hex_to_bytes(merge_pair[1]);
        merges_.push_back({a, b});
    }
}