#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>

// CommandSchema — schema_'ya atanan veri yapısı
struct CommandSchema
{
    std::string command;
    std::vector<std::string> params;
    std::string description;
};

class CommandParamExtractor
{
public:
    // Orijinal constructor — 3 parametre
    // schema_json: JSON string (inline). Boş bırakılırsa parse yapılmaz.
    CommandParamExtractor(
        const std::unordered_map<std::string, std::vector<float>> &embeddings_map,
        const std::string &schema_json,
        bool debug = false);

    // Tokenize (word-level; BPE gelecekte eklenebilir)
    std::vector<std::string> tokenize(const std::string &sentence,
                                      bool use_bpe = false);

    // Parametre çıkar
    std::vector<std::pair<std::string, float>>
    extract(const std::string &sentence, const std::string &command);

    // Embedding lookup (token → embedding vektörü)
    const std::vector<float> *get_embedding(const std::string &token) const;

    // İstatistik string'i
    std::string get_stats() const;

    // Debug çıktısı
    void dump_embeddings(int max_tokens = 20) const;

private:
    static constexpr float EPSILON = 1e-8f;

    const std::unordered_map<std::string, std::vector<float>> &embeddings_;
    std::unordered_map<std::string, CommandSchema> schema_;
    bool debug_mode_;

    float cosine_similarity(const std::vector<float> &a,
                            const std::vector<float> &b) const;
    std::string normalize(const std::string &s) const;
    std::vector<std::string> split_words(const std::string &s) const;
    void parse_schema(const std::string &json_str);
};