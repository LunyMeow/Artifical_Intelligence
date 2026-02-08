#ifndef BYTE_BPE_TOKENIZER_H
#define BYTE_BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

class ByteBPETokenizer {
public:
    using ByteSeq = std::vector<uint8_t>;
    using BytePair = std::pair<ByteSeq, ByteSeq>;
    
    bool debugLog = true; // Debug modunu kontrol eden değişken
    
    ByteBPETokenizer(int vocab_size = 16000);
    
    // Training
    int train(const std::vector<std::string>& texts);
    
    // Encoding/Decoding
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    
    // Save/Load
    void save(const std::string& path);
    bool load(const std::string& path);
    
    // Public getters for embeddings
    const std::map<int, ByteSeq>& get_id_to_token() const {
        return id_to_token_;
    }
    
private:
    int vocab_size_;
    std::map<ByteSeq, int> token_to_id_;
    std::map<int, ByteSeq> id_to_token_;
    std::vector<BytePair> merges_;
    
    // Helper functions
    static ByteSeq byte_encode(const std::string& text);
    static std::map<BytePair, int> get_pair_counts(
        const std::vector<std::vector<ByteSeq>>& sequences
    );
    static std::pair<std::vector<std::vector<ByteSeq>>, ByteSeq> merge_pair(
        const BytePair& pair,
        const std::vector<std::vector<ByteSeq>>& sequences
    );
    static std::string bytes_to_hex(const ByteSeq& bytes);
    static ByteSeq hex_to_bytes(const std::string& hex);
};

#endif // BYTE_BPE_TOKENIZER_H