#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// ByteBPETokenizer forward declaration
class ByteBPETokenizer;
class CommandParamExtractor;

// TokenizerMode — Buildv1_3_2.cpp'de tanımlı.
// InferenceEngine.h bu dosyadan SONRA include edildiği için
// TokenizerMode zaten mevcut; burada yeniden tanımlamamak için guard kullan.
#ifndef TOKENIZER_MODE_DEFINED
#define TOKENIZER_MODE_DEFINED
enum class TokenizerMode
{
    WORD    = 0,
    SUBWORD = 1,
    BPE     = 2
};
#endif

// ExtractedCommand struct
struct ExtractedCommand
{
    std::string command;
    std::vector<std::string> tokenizedSentence;
    std::vector<std::pair<std::string, float>> parameters;
    bool debug = false;
};

class InferenceEngine
{
public:
    // ============================================================
    // FIX-1: Constructor — TokenizerMode ve ByteBPETokenizer* eklendi
    // (default parametreler geriye uyumluluğu korur)
    // ============================================================
    InferenceEngine(
        CommandParamExtractor *param_extractor,
        const std::unordered_map<std::string, std::vector<float>> &word_embeddings,
        const std::unordered_map<std::string, std::vector<float>> &command_embeddings,
        bool debug = false,
        TokenizerMode tokenizer_mode = TokenizerMode::WORD,
        ByteBPETokenizer *bpe_tokenizer = nullptr);

    // Schema yükle
    bool init_with_schema(const std::string &schema_path);

    // Ana inference fonksiyonu
    ExtractedCommand infer(const std::string &sentence,
                           const std::string &command_name,
                           bool debug = false);

    // Token-by-token generation (FIX-2: top-k + hard-exclude)
    std::string generate_tokens(const std::string &sentence,
                                const std::string &start_command,
                                int max_tokens = 20);

    // Komut + parametre birleştir
    std::string generate_command(const std::string &sentence,
                                 const std::string &command_name,
                                 int max_tokens = 10);

    // Template doldur
    std::string fill_template(
        const std::string &command,
        const std::vector<std::pair<std::string, float>> &parameters);

    // En yakın komutu bul
    std::string predict_command(const std::vector<float> &input_embedding);

    // Debug raporu
    std::string get_debug_report() const;

private:
    // ── Dependencies ────────────────────────────────────────────
    CommandParamExtractor *param_extractor_;

    // ── Embedding maps ──────────────────────────────────────────
    // FIX-3: word_embeddings_ artık allEmbeddings (word+command) olarak geliyor
    std::unordered_map<std::string, std::vector<float>> word_embeddings_;
    std::unordered_map<std::string, std::vector<float>> command_embeddings_;

    // ── FIX-1: Tokenizer state ──────────────────────────────────
    TokenizerMode  tokenizer_mode_ = TokenizerMode::WORD;
    ByteBPETokenizer *bpe_tokenizer_ = nullptr;

    // ── Template ve schema map'leri ─────────────────────────────
    std::unordered_map<std::string, std::vector<std::string>> command_templates_;
    std::unordered_map<std::string, std::string>              command_descriptions_;

    // ── Debug ────────────────────────────────────────────────────
    bool debug_mode_ = false;
    std::string last_debug_report_;

    // ── Private helpers ─────────────────────────────────────────
    void init_templates();
    bool load_command_schema(const std::string &schema_path);
    std::string normalize(const std::string &s) const;
    float cosine_similarity(const std::vector<float> &a,
                            const std::vector<float> &b) const;

    // FIX-1: BPE/SUBWORD/WORD modunu destekler
    std::vector<std::string> tokenize_sentence(const std::string &sentence);

    std::vector<std::pair<std::string, float>>
    extract_parameters(const std::string &sentence,
                       const std::string &command_name);

    std::vector<std::pair<std::string, float>>
    extract_parameters_from_sentence(const std::string &sentence,
                                     const std::vector<std::string> &placeholders);
};