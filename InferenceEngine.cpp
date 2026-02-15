#include "InferenceEngine.h"
#include "ByteBPE/ByteBPETokenizer.h"
#include "CommandParamExtractor.h"
#include <sstream>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>    // FIX-2: top-k örnekleme için
#include <deque>     // FIX-2: hard-exclude sliding window
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ============================================================
// FIX-1: Constructor — TokenizerMode ve ByteBPETokenizer* eklendi
// ============================================================
InferenceEngine::InferenceEngine(
    CommandParamExtractor *param_extractor,
    const std::unordered_map<std::string, std::vector<float>> &word_embeddings,
    const std::unordered_map<std::string, std::vector<float>> &command_embeddings,
    bool debug,
    TokenizerMode tokenizer_mode,
    ByteBPETokenizer *bpe_tokenizer)
    : param_extractor_(param_extractor),
      word_embeddings_(word_embeddings),
      command_embeddings_(command_embeddings),
      debug_mode_(debug),
      tokenizer_mode_(tokenizer_mode),
      bpe_tokenizer_(bpe_tokenizer)
{
    if (!param_extractor_)
    {
        std::cerr << "[InferenceEngine] ERROR: param_extractor is null!\n";
    }

    // Otomatik schema yükleme: LLM/Embeddings/command_schema.json
    // Bulunamazsa command_templates_ boş kalır (fill_template "no template" döner)
    const std::string default_schema = "LLM/Embeddings/command_schema.json";
    bool loaded = load_command_schema(default_schema);
    if (debug_mode_)
    {
        std::cout << "[InferenceEngine] Schema: "
                  << (loaded
                      ? "Loaded from " + default_schema
                      : "Not found (" + default_schema + ") — templates empty")
                  << "\n";
    }
}

void InferenceEngine::init_templates()
{
    // Tüm komut şablonları LLM/Embeddings/command_schema.json'dan yüklenir.
    // Constructor otomatik çağırır; bu fonksiyon artık hardcode tanım içermez.
    // Dosya bulunamazsa command_templates_ boş kalır ve fill_template()
    // komutu ham (parametresiz) döndürür.
    if (debug_mode_)
        std::cout << "[InferenceEngine] init_templates: JSON-only mode\n";
}

bool InferenceEngine::load_command_schema(const std::string &schema_path)
{
    try
    {
        std::ifstream schema_file(schema_path);
        if (!schema_file.good())
        {
            if (debug_mode_)
                std::cerr << "[load_command_schema] File not found: " << schema_path << "\n";
            return false;
        }

        json schema = json::parse(schema_file);

        // Clear existing templates
        command_templates_.clear();
        command_descriptions_.clear();

        // Load from JSON
        for (auto &[cmd_name, cmd_data] : schema.items())
        {
            try
            {
                std::vector<std::string> params;
                
                if (cmd_data.contains("params"))
                {
                    for (const auto &param : cmd_data["params"])
                    {
                        params.push_back(param.get<std::string>());
                    }
                }

                command_templates_[cmd_name] = params;

                if (cmd_data.contains("description"))
                {
                    command_descriptions_[cmd_name] = cmd_data["description"].get<std::string>();
                }

                if (debug_mode_)
                {
                    std::cout << "[load_command_schema] Loaded: " << cmd_name 
                              << " [" << params.size() << " params]\n";
                }
            }
            catch (const std::exception &e)
            {
                if (debug_mode_)
                    std::cerr << "[load_command_schema] Error parsing command " << cmd_name 
                              << ": " << e.what() << "\n";
                continue;
            }
        }

        if (debug_mode_)
        {
            std::cout << "[load_command_schema] Successfully loaded " 
                      << command_templates_.size() << " commands\n";
        }

        return true;
    }
    catch (const std::exception &e)
    {
        if (debug_mode_)
            std::cerr << "[load_command_schema] JSON parse error: " << e.what() << "\n";
        return false;
    }
}

std::string InferenceEngine::normalize(const std::string &s) const
{
    std::string result = s;
    // Convert to lowercase
    for (char &c : result)
        c = std::tolower(static_cast<unsigned char>(c));
    // Trim leading/trailing whitespace
    size_t start = result.find_first_not_of(" \t\r\n");
    size_t end = result.find_last_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    return result.substr(start, end - start + 1);
}

float InferenceEngine::cosine_similarity(const std::vector<float> &a,
                                        const std::vector<float> &b) const
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return -2.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-8f)
        return 0.0f;

    return dot / denom;
}

// ============================================================
// FIX-1: tokenize_sentence — BPE/SUBWORD/WORD modunu destekler
// sentence_embedding() içindeki tokenize mantığıyla birebir aynı.
// ============================================================
std::vector<std::string> InferenceEngine::tokenize_sentence(const std::string &sentence)
{
    std::vector<std::string> tokens;

    // ── BPE modu ────────────────────────────────────────────────────────────
    if (tokenizer_mode_ == TokenizerMode::BPE)
    {
        if (!bpe_tokenizer_)
        {
            // BPE tokenizer yoksa WORD moduna düş
            if (debug_mode_)
                std::cerr << "[tokenize_sentence] BPE tokenizer null — WORD moduna dusuldu\n";
        }
        else
        {
            // Kelimelere ayır
            std::vector<std::string> words;
            size_t i = 0, n = sentence.size();
            while (i < n)
            {
                while (i < n && sentence[i] == ' ') i++;
                size_t start = i;
                while (i < n && sentence[i] != ' ') i++;
                if (start < i)
                    words.push_back(sentence.substr(start, i - start));
            }

            for (const auto &word : words)
            {
                // Özel token (<dir>, <file> vb.) — doğrudan ekle
                if (word.size() > 2 && word.front() == '<' && word.back() == '>')
                {
                    tokens.push_back(word);
                    continue;
                }

                // BPE encode
                auto ids = bpe_tokenizer_->encode(word);

                if (ids.empty())
                {
                    tokens.push_back(word); // fallback: kelimenin kendisi
                    continue;
                }

                for (int id : ids)
                {
                    auto decoded = bpe_tokenizer_->decode({id});
                    if (!decoded.empty())
                        tokens.push_back(decoded);
                }
            }

            if (debug_mode_)
            {
                std::cout << "[Tokenize] Input: \"" << sentence << "\"\n";
                std::cout << "[Tokenize] BPE tokens: [";
                for (const auto &t : tokens) std::cout << "\"" << t << "\" ";
                std::cout << "]\n";
            }
            return tokens;
        }
    }

    // ── SUBWORD modu (char n-gram) ───────────────────────────────────────────
    if (tokenizer_mode_ == TokenizerMode::SUBWORD)
    {
        constexpr int subword_n = 3;
        for (size_t i = 0; i + subword_n <= sentence.size(); i++)
        {
            bool valid = true;
            for (int j = 0; j < subword_n; j++)
                if (sentence[i + j] == ' ') { valid = false; break; }
            if (valid)
                tokens.emplace_back(sentence.substr(i, subword_n));
        }

        if (debug_mode_)
        {
            std::cout << "[Tokenize] Input: \"" << sentence << "\"\n";
            std::cout << "[Tokenize] SUBWORD tokens: [";
            for (const auto &t : tokens) std::cout << "\"" << t << "\" ";
            std::cout << "]\n";
        }
        return tokens;
    }

    // ── WORD modu (fallback, param_extractor kullanır) ──────────────────────
    if (!param_extractor_)
    {
        std::cerr << "[tokenize_sentence] param_extractor is null!\n";
        return tokens;
    }

    tokens = param_extractor_->tokenize(sentence, false);

    if (debug_mode_)
    {
        std::cout << "[Tokenize] Input: \"" << sentence << "\"\n";
        std::cout << "[Tokenize] Word tokens: [";
        for (const auto &t : tokens) std::cout << "\"" << t << "\" ";
        std::cout << "]\n";
    }
    return tokens;
}

std::vector<std::pair<std::string, float>>
InferenceEngine::extract_parameters(const std::string &sentence,
                                   const std::string &command_name)
{
    if (!param_extractor_)
    {
        std::cerr << "[extract_parameters] param_extractor is null!\n";
        return {};
    }

    auto params = param_extractor_->extract(sentence, command_name);

    if (debug_mode_)
    {
        std::cout << "[extract_parameters] Command: \"" << command_name << "\"\n";
        std::cout << "[extract_parameters] Sentence: \"" << sentence << "\"\n";
        std::cout << "[extract_parameters] Extracted " << params.size() << " parameters:\n";
        for (const auto &[token, score] : params)
        {
            std::cout << "  - \"" << token << "\" (score=" << score << ")\n";
        }
    }

    return params;
}

std::string InferenceEngine::fill_template(
    const std::string &command,
    const std::vector<std::pair<std::string, float>> &parameters)
{
    // Find template for this command
    auto it = command_templates_.find(command);
    if (it == command_templates_.end())
    {
        if (debug_mode_)
            std::cout << "[fill_template] No template for command: \"" << command << "\"\n";
        return command; // Return command without parameters
    }

    std::string result = command;
    const auto &placeholders = it->second;

    // Fill each placeholder with extracted parameters
    for (size_t i = 0; i < placeholders.size() && i < parameters.size(); i++)
    {
        result += " " + parameters[i].first; // Append parameter value
    }

    if (debug_mode_)
    {
        std::cout << "[fill_template] Filled command: \"" << result << "\"\n";
    }

    return result;
}

/**
 * @brief Parametreleri cümleden semantic similarity ile çıkart
 * Örnek: "yedekleri backup/ klasörüne kopyala" → ["yedekleri", "backup/"]
 */
std::vector<std::pair<std::string, float>>
InferenceEngine::extract_parameters_from_sentence(
    const std::string &sentence,
    const std::vector<std::string> &placeholders)
{
    if (debug_mode_)
    {
        std::cout << "[extract_parameters_from_sentence] Sentence: \"" << sentence << "\"\n";
        std::cout << "[extract_parameters_from_sentence] Placeholders: ";
        for (const auto &p : placeholders) std::cout << p << " ";
        std::cout << "\n";
        std::cout << "[extract_parameters_from_sentence] Word embeddings size: " << word_embeddings_.size() << "\n";
    }

    std::vector<std::pair<std::string, float>> result;
    
    // Cümleyi tokenize et
    auto tokens = tokenize_sentence(sentence);
    
    // Eğer embeddings yüklü değilse, direkt token'ları kullan
    if (word_embeddings_.empty())
    {
        if (debug_mode_)
            std::cout << "[extract_parameters_from_sentence] WARNING: word_embeddings is empty!\n";
        
        // Fallback: parametreleri kelimeleri olarak döndür
        // İlk N kelimeyi parametreler olarak kabul et
        for (size_t i = 0; i < placeholders.size() && i < tokens.size(); i++)
        {
            result.push_back({tokens[i], 0.5f});
        }
        
        return result;
    }
    
    if (debug_mode_)
    {
        std::cout << "[extract_parameters_from_sentence] Tokens: ";
        for (const auto &t : tokens) std::cout << "\"" << t << "\" ";
        std::cout << "\n";
    }

    // Her placeholder için en benzer token'ı bul
    for (size_t placeholder_idx = 0; placeholder_idx < placeholders.size(); placeholder_idx++)
    {
        const auto& placeholder = placeholders[placeholder_idx];
        
        float best_similarity = -1e9f;
        std::string best_token;
        int best_token_idx = -1;
        
        // Placeholder tipini belirle (örn: "<FILE>" → "file")
        std::string placeholder_type = placeholder;
        if (placeholder_type.front() == '<') placeholder_type.erase(0, 1);
        if (placeholder_type.back() == '>') placeholder_type.pop_back();
        std::transform(placeholder_type.begin(), placeholder_type.end(), 
                      placeholder_type.begin(), ::tolower);
        
        if (debug_mode_)
            std::cout << "[extract_parameters_from_sentence] Looking for: " << placeholder_type << "\n";

        // Strategy 1: Position-based selection (for Turkish command structure)
        // Usually: param1 param2 ... command
        // So first few tokens are likely parameters
        
        // Expected token index for this placeholder
        // Placeholder 0 (<SRC>) → token 0
        // Placeholder 1 (<DST>) → token 1, etc.
        int expected_idx = placeholder_idx;
        
        if (expected_idx < (int)tokens.size())
        {
            // Try position-based approach first
            const auto& token = tokens[expected_idx];
            
            // Check if token exists in embeddings
            auto it_word = word_embeddings_.find(token);
            if (it_word != word_embeddings_.end())
            {
                float position_score = 0.8f;  // High score for position match
                
                // Apply type-specific heuristics on top of position
                if (placeholder_type == "file" || placeholder_type == "src" || placeholder_type == "source")
                {
                    // File-like: not containing / and not ending with /
                    if (token.find('/') == std::string::npos && token.back() != '/')
                        position_score += 0.1f;  // Bonus for not being a directory
                }
                else if (placeholder_type == "dir" || placeholder_type == "path" || placeholder_type == "dst" || placeholder_type == "destination")
                {
                    // Directory-like: containing / or ending with /
                    if (token.find('/') != std::string::npos || token.back() == '/')
                        position_score += 0.1f;  // Bonus for being a directory
                }
                
                best_similarity = position_score;
                best_token = token;
                best_token_idx = expected_idx;
                
                if (debug_mode_)
                    std::cout << "[extract_parameters_from_sentence]   Position match: \"" << token 
                              << "\" at index " << expected_idx << " (score=" << position_score << ")\n";
            }
        }
        
        // Strategy 2: If position-based didn't work, scan all tokens
        if (best_token.empty())
        {
            for (size_t token_idx = 0; token_idx < tokens.size(); token_idx++)
            {
                const auto& token = tokens[token_idx];
                
                // Skip if already used
                bool already_used = false;
                for (const auto& [used_param, _] : result)
                {
                    if (used_param == token)
                    {
                        already_used = true;
                        break;
                    }
                }
                if (already_used) continue;
                
                // Word embedding'lerde ara
                auto it_word = word_embeddings_.find(token);
                if (it_word == word_embeddings_.end())
                    continue;

                // Token embedding'ini al
                const auto &token_emb = it_word->second;

                // Type-specific scoring
                float semantic_score = 0.0f;

                if (placeholder_type == "file")
                {
                    // File-like: uzantılı olup, / içermiyor
                    semantic_score = (token.find('.') != std::string::npos) ? 0.8f : 0.0f;
                    if (token.find('/') == std::string::npos && token.find('.') == std::string::npos)
                        semantic_score = 0.5f;
                }
                else if (placeholder_type == "dir" || placeholder_type == "path")
                {
                    // Directory-like: / içeriyor veya çok uzun
                    semantic_score = (token.find('/') != std::string::npos) ? 0.9f : 0.0f;
                    if (token.length() > 5)
                        semantic_score = std::max(semantic_score, 0.6f);
                }
                else if (placeholder_type == "src" || placeholder_type == "source")
                {
                    // Source: file-like
                    semantic_score = (token.find('/') == std::string::npos && token.find('.') != std::string::npos) ? 0.8f : 0.5f;
                }
                else if (placeholder_type == "dst" || placeholder_type == "destination")
                {
                    // Destination: directory or later position
                    semantic_score = (token.find('/') != std::string::npos) ? 0.8f : 0.3f;
                }
                else if (placeholder_type == "perms")
                {
                    // Permissions: sayısal veya 'rwx' benzeri
                    semantic_score = (token.length() > 0 && isdigit(token[0])) ? 0.9f : 0.0f;
                    if (token.find('r') != std::string::npos || token.find('w') != std::string::npos)
                        semantic_score = 0.9f;
                }
                else if (placeholder_type == "owner" || placeholder_type == "user")
                {
                    // Owner: kullanıcı adı, genellikle kısa
                    semantic_score = (token.length() <= 20 && token.find('/') == std::string::npos) ? 0.7f : 0.0f;
                }
                else if (placeholder_type == "pattern")
                {
                    // Pattern: wildcard veya regex
                    semantic_score = (token.find('*') != std::string::npos || token.find('?') != std::string::npos) ? 0.9f : 0.5f;
                }

                if (semantic_score > best_similarity)
                {
                    best_similarity = semantic_score;
                    best_token = token;
                    best_token_idx = token_idx;

                    if (debug_mode_)
                        std::cout << "[extract_parameters_from_sentence]   Candidate: \"" << token 
                                  << "\" at index " << token_idx << " (score=" << semantic_score << ")\n";
                }
            }
        }

        if (!best_token.empty() && best_similarity > 0.1f)
        {
            result.push_back({best_token, best_similarity});
            
            if (debug_mode_)
                std::cout << "[extract_parameters_from_sentence] Selected for " << placeholder 
                          << ": \"" << best_token << "\" (score=" << best_similarity << ")\n";
        }
        else
        {
            if (debug_mode_)
                std::cout << "[extract_parameters_from_sentence] No good match for " << placeholder << "\n";
        }
    }

    return result;
}
std::string InferenceEngine::predict_command(const std::vector<float> &input_embedding)
{
    std::string best_cmd;
    float best_sim = -1e9f;

    for (const auto &[cmd, emb] : command_embeddings_)
    {
        float sim = cosine_similarity(input_embedding, emb);
        if (sim > best_sim)
        {
            best_sim = sim;
            best_cmd = cmd;
        }
    }

    if (debug_mode_)
    {
        std::cout << "[predict_command] Best match: \"" << best_cmd
                  << "\" (similarity=" << best_sim << ")\n";
    }

    return best_cmd;
}

ExtractedCommand InferenceEngine::infer(const std::string &sentence,
                                       const std::string &command_name,
                                       bool debug)
{
    ExtractedCommand result;
    result.command = command_name;
    result.debug = debug || debug_mode_;

    std::ostringstream debug_log;

    if (result.debug)
        debug_log << "[infer] Starting inference...\n";

    // Step 1: Tokenize
    result.tokenizedSentence = tokenize_sentence(sentence);
    if (result.debug)
    {
        debug_log << "[infer] Tokenized: [";
        for (size_t i = 0; i < result.tokenizedSentence.size(); i++)
        {
            debug_log << "\"" << result.tokenizedSentence[i] << "\"";
            if (i < result.tokenizedSentence.size() - 1)
                debug_log << ", ";
        }
        debug_log << "]\n";
    }

    // Step 2: Extract parameters using semantic similarity to template placeholders
    // Get the placeholders for this command
    auto template_it = command_templates_.find(command_name);
    if (template_it != command_templates_.end())
    {
        // Use semantic matching based on placeholder types
        result.parameters = extract_parameters_from_sentence(sentence, template_it->second);

        if (result.debug)
        {
            debug_log << "[infer] Extracted " << result.parameters.size() 
                     << " parameters using semantic similarity\n";
            for (const auto &[token, score] : result.parameters)
            {
                debug_log << "  [PARAM] \"" << token << "\" (confidence=" << score << ")\n";
            }
        }
    }
    else
    {
        // Fallback: use old method if no template
        result.parameters = extract_parameters(sentence, command_name);

        if (result.debug)
        {
            debug_log << "[infer] Extracted " << result.parameters.size() 
                     << " parameters using fallback method\n";
            for (const auto &[token, score] : result.parameters)
            {
                debug_log << "  [PARAM] \"" << token << "\" (confidence=" << score << ")\n";
            }
        }
    }

    last_debug_report_ = debug_log.str();

    return result;
}

std::string InferenceEngine::generate_command(const std::string &sentence,
                                            const std::string &command_name,
                                            int max_tokens)
{
    if (max_tokens < 1)
        max_tokens = 10;

    if (debug_mode_)
    {
        std::cout << "[generate_command] Input: \"" << sentence << "\"\n";
        std::cout << "[generate_command] Command: \"" << command_name << "\"\n";
        std::cout << "[generate_command] Max tokens: " << max_tokens << "\n";
    }

    // Run inference pipeline
    auto extracted = infer(sentence, command_name, debug_mode_);

    // Fill template with extracted parameters
    std::string filled_command = fill_template(command_name, extracted.parameters);

    if (debug_mode_)
    {
        std::cout << "[generate_command] Final output: \"" << filled_command << "\"\n";
    }

    return filled_command;
}

std::string InferenceEngine::get_debug_report() const
{
    return last_debug_report_;
}

bool InferenceEngine::init_with_schema(const std::string &schema_path)
{
    if (debug_mode_)
        std::cout << "[init_with_schema] Loading schema from: " << schema_path << "\n";

    // Mevcut template'leri temizle ve yeniden yükle
    command_templates_.clear();
    command_descriptions_.clear();

    bool success = load_command_schema(schema_path);

    if (!success && debug_mode_)
    {
        std::cout << "[init_with_schema] Failed to load schema: " << schema_path
                  << "\n  → Templates cleared. fill_template() will return raw command.\n";
    }

    if (debug_mode_)
    {
        std::cout << "[init_with_schema] Total commands loaded: "
                  << command_templates_.size() << "\n";
    }

    return success;
}

/**
 * FIX-2: generate_tokens — top-k örnekleme + hard-exclude + BPE-aware context
 *
 * Eski implementasyonun sorunları:
 *  1. Greedy seçim → aynı token tekrar tekrar seçiliyordu (sh, sh, sh)
 *  2. Alpha decay context'i hızla tek bir token'a yakınsatıyordu
 *  3. Tekrar tespiti geç devreye giriyordu (3/3 ardışık)
 *  4. BPE tokenizer yokken context embedding sıfır kalıyordu
 *
 * Yeni yaklaşım:
 *  - Top-k (k=5) örnekleme: her adımda en iyi k aday arasından uniform seç
 *  - Hard-exclude deque: son MAX_EXCLUDE token tekrar seçilemiyor
 *  - Sabit küçük alpha (0.05): context kilitlenmiyor
 *  - start_command embedding context'e dahil: başlangıç noktası anlamlı
 */
std::string InferenceEngine::generate_tokens(const std::string &sentence,
                                            const std::string &start_command,
                                            int max_tokens)
{
    if (max_tokens < 1) max_tokens = 20;

    constexpr int   TOP_K        = 5;    // top-k adaydan rastgele seç
    constexpr int   MAX_EXCLUDE  = 4;    // son N token'ı hard-exclude et
    constexpr float ALPHA        = 0.05f; // context güncelleme oranı (sabit, küçük)
    constexpr float MIN_SIM      = 0.05f; // bu altında üretimi durdur

    static std::mt19937 rng(std::random_device{}());

    if (debug_mode_)
    {
        std::cout << "\n[generate_tokens] Starting token generation\n"
                  << "[generate_tokens] Input: \"" << sentence << "\"\n"
                  << "[generate_tokens] Start command: \"" << start_command << "\"\n"
                  << "[generate_tokens] Max tokens: " << max_tokens << "\n";
    }

    std::string output = start_command;
    if (!start_command.empty()) output += " ";

    // ── Başlangıç context: sentence token'larının ortalaması ──────────────────
    std::vector<float> context(50, 0.0f);
    {
        auto input_tokens = tokenize_sentence(sentence);

        if (debug_mode_)
        {
            std::cout << "[generate_tokens] Input tokens: ";
            for (const auto &t : input_tokens) std::cout << "\"" << t << "\" ";
            std::cout << "\n";
        }

        int found = 0;
        for (const auto &tok : input_tokens)
        {
            auto it = word_embeddings_.find(tok);
            if (it != word_embeddings_.end())
            {
                for (int i = 0; i < 50; i++) context[i] += it->second[i];
                found++;
            }
        }

        if (found > 0)
        {
            float inv = 1.0f / found;
            for (float &v : context) v *= inv;
        }
        else
        {
            // FIX: sıfır yerine start_command embedding'ini kullan
            auto it_cmd = command_embeddings_.find(start_command);
            if (it_cmd != command_embeddings_.end())
            {
                context = it_cmd->second;
                if (debug_mode_)
                    std::cout << "[generate_tokens] No input tokens found — "
                                 "using start_command embedding as context\n";
            }
            else
            {
                if (debug_mode_)
                    std::cout << "[generate_tokens] WARNING: No input tokens AND "
                                 "start_command not in embeddings — context is zero\n";
            }
        }

        // start_command embedding'ini context'e hafifçe karıştır (%20)
        {
            auto it_start = command_embeddings_.find(start_command);
            if (it_start != command_embeddings_.end())
            {
                for (int i = 0; i < 50; i++)
                    context[i] = 0.80f * context[i] + 0.20f * it_start->second[i];
            }
        }
    }

    // ── Üretim döngüsü ───────────────────────────────────────────────────────
    int token_count = 0;
    std::deque<std::string> recent; // hard-exclude sliding window

    while (token_count < max_tokens)
    {
        // ── Aday listesi: hard-exclude uygulanmış, MIN_SIM üzeri ─────────────
        std::vector<std::pair<float, std::string>> candidates;
        candidates.reserve(command_embeddings_.size());

        for (const auto &[tok, emb] : command_embeddings_)
        {
            // Hard-exclude: deque'daki son MAX_EXCLUDE token seçilemiyor
            bool excluded = false;
            for (const auto &r : recent)
                if (r == tok) { excluded = true; break; }
            if (excluded) continue;

            float sim = cosine_similarity(context, emb);
            if (sim >= MIN_SIM)
                candidates.push_back({sim, tok});
        }

        if (candidates.empty())
        {
            if (debug_mode_)
                std::cout << "[generate_tokens] No candidates above threshold. Stopping.\n";
            break;
        }

        // En yüksek benzerlikten sırala, top-k al
        std::sort(candidates.rbegin(), candidates.rend());
        int k = std::min(TOP_K, (int)candidates.size());
        candidates.resize(k);

        // Uniform rastgele seçim (kilitlenmeyi önler)
        std::uniform_int_distribution<int> dist(0, k - 1);
        int chosen_idx = dist(rng);
        const std::string &best_token = candidates[chosen_idx].second;
        float best_sim = candidates[chosen_idx].first;

        if (debug_mode_)
        {
            std::cout << "[generate_tokens] Token " << (token_count + 1)
                      << "/" << max_tokens << ": \"" << best_token
                      << "\" (similarity=" << best_sim << ", top-k=" << k << ")\n";
        }

        // <end> token kontrolü
        {
            std::string tok_lower = best_token;
            std::transform(tok_lower.begin(), tok_lower.end(), tok_lower.begin(), ::tolower);
            if (tok_lower == "<end>")
            {
                if (debug_mode_)
                    std::cout << "[generate_tokens] End token detected. Stopping.\n";
                break;
            }
        }

        // Çıktıya ekle
        output += best_token;
        if (token_count < max_tokens - 1) output += " ";

        // Hard-exclude güncelle
        recent.push_back(best_token);
        if ((int)recent.size() > MAX_EXCLUDE)
            recent.pop_front();

        // Context güncelle — sabit küçük alpha
        {
            auto it_cmd = command_embeddings_.find(best_token);
            if (it_cmd != command_embeddings_.end())
            {
                for (int i = 0; i < 50; i++)
                    context[i] = (1.0f - ALPHA) * context[i] + ALPHA * it_cmd->second[i];
            }
        }

        token_count++;
    }

    // Sondaki boşlukları temizle
    while (!output.empty() && output.back() == ' ')
        output.pop_back();

    if (debug_mode_)
    {
        std::cout << "[generate_tokens] Generation complete\n"
                  << "[generate_tokens] Final output: \"" << output << "\"\n"
                  << "[generate_tokens] Tokens generated: " << token_count << "\n\n";
    }

    return output;
}