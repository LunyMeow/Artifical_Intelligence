#include "CommandParamExtractor.h"
#include <sstream>
#include <cctype>
#include <iomanip>
#include <cstring>
// Needed for std::remove_if, std::sort, std::sqrt
#include <algorithm>
#include <cmath>

CommandParamExtractor::CommandParamExtractor(
    const std::unordered_map<std::string, std::vector<float>> &embeddings_map,
    const std::string &schema_json,
    bool debug)
    : embeddings_(embeddings_map), debug_mode_(debug)
{
    if (debug_mode_)
    {
        std::cout << "[CommandParamExtractor] Initializing with "
                  << embeddings_map.size() << " embeddings\n";
    }

    // Parse command schema if provided
    if (!schema_json.empty())
    {
        parse_schema(schema_json);
    }
}

float CommandParamExtractor::cosine_similarity(const std::vector<float> &a,
                                              const std::vector<float> &b) const
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return -2.0f; // Invalid

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < EPSILON)
        return 0.0f;

    return dot / denom;
}

std::string CommandParamExtractor::normalize(const std::string &s) const
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

std::vector<std::string> CommandParamExtractor::split_words(const std::string &s) const
{
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string word;
    while (iss >> word)
    {
        // Remove punctuation (keep < and > for tags like <DIR>, <FILE>)
        word.erase(std::remove_if(word.begin(), word.end(),
                                 [](char c) { 
                                     return std::ispunct(static_cast<unsigned char>(c)) && c != '<' && c != '>'; 
                                 }),
                  word.end());
        if (!word.empty())
            result.push_back(word);
    }
    return result;
}

std::vector<std::string>
CommandParamExtractor::tokenize(const std::string &sentence, bool use_bpe)
{
    std::string normalized = normalize(sentence);
    std::vector<std::string> tokens = split_words(normalized);

    if (debug_mode_)
    {
        std::cout << "[Tokenize] Input: \"" << sentence << "\"\n";
        std::cout << "[Tokenize] Normalized: \"" << normalized << "\"\n";
        std::cout << "[Tokenize] Word tokens: [";
        for (size_t i = 0; i < tokens.size(); i++)
        {
            std::cout << "\"" << tokens[i] << "\"";
            if (i < tokens.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // TODO: Implement BPE tokenization if use_bpe=true
    // For now, return word-level tokens

    return tokens;
}

const std::vector<float> *
CommandParamExtractor::get_embedding(const std::string &token) const
{
    auto it = embeddings_.find(token);
    if (it == embeddings_.end())
    {
        // Eğer token bulunamadıysa, tag'i kaldırarak dene: <dir> → dir
        std::string token_without_tags = token;
        if (token.length() >= 2 && token.front() == '<' && token.back() == '>')
        {
            token_without_tags = token.substr(1, token.length() - 2);
            
            auto it2 = embeddings_.find(token_without_tags);
            if (it2 != embeddings_.end())
            {
                if (debug_mode_)
                    std::cout << "[DEBUG] Token found (tag removed): \"" << token_without_tags 
                              << "\" from \"" << token << "\" (" << it2->second.size() << "-dim)\n";
                return &it2->second;
            }
        }
        
        if (debug_mode_)
            std::cout << "[DEBUG] Token NOT in DB: \"" << token << "\"\n";
        return nullptr;
    }

    if (debug_mode_)
    {
        std::cout << "[DEBUG] Token found: \"" << token << "\" "
                  << "(" << it->second.size() << "-dim)\n";
    }

    return &it->second;
}

std::vector<std::pair<std::string, float>>
CommandParamExtractor::extract(const std::string &sentence, const std::string &command)
{
    std::vector<std::pair<std::string, float>> result;

    if (debug_mode_)
    {
        std::cout << "\n[Extract] Command: \"" << command << "\"\n";
        std::cout << "[Extract] Sentence: \"" << sentence << "\"\n";
    }

    // Tokenize sentence
    std::vector<std::string> tokens = tokenize(sentence);

    if (tokens.empty())
    {
        if (debug_mode_)
            std::cout << "[Extract] No tokens extracted!\n";
        return result;
    }

    // Get command embedding (for scoring context)
    const std::vector<float> *cmd_emb = get_embedding(command);

    // Score each token
    std::vector<std::pair<std::string, float>> scored;

    for (const auto &token : tokens)
    {
        const std::vector<float> *token_emb = get_embedding(token);

        float score = 0.0f;

        if (token_emb && cmd_emb)
        {
            // Similarity to command embedding
            score = cosine_similarity(*token_emb, *cmd_emb);
        }
        else if (!token_emb)
        {
            // Unknown token = potential parameter
            score = -1.0f;
        }
        else
        {
            score = 0.0f;
        }

        if (debug_mode_)
        {
            std::cout << "[Score] \"" << token << "\" → " << score << "\n";
        }

        scored.push_back({token, score});
    }

    // Sort by score (ascending = lowest score = most likely parameter)
    std::sort(scored.begin(), scored.end(),
             [](const auto &a, const auto &b)
             { return a.second < b.second; });

    // Determine number of parameters needed
    size_t param_count = 1; // Default
    auto schema_it = schema_.find(command);
    if (schema_it != schema_.end())
    {
        param_count = schema_it->second.params.size();
    }

    // Return top N tokens as parameters
    for (size_t i = 0; i < scored.size() && result.size() < param_count; i++)
    {
        result.push_back(scored[i]);
    }

    if (debug_mode_)
    {
        std::cout << "[Extract] Extracted " << result.size() << " parameters:\n";
        for (const auto &[tok, score] : result)
        {
            std::cout << "  - \"" << tok << "\" (score=" << score << ")\n";
        }
    }

    return result;
}

void CommandParamExtractor::parse_schema(const std::string &json_str)
{
    // Simplified JSON parsing (no external library)
    // Expected format: {"command_name": {"params": ["<PARAM1>", "<PARAM2>"]}}

    if (debug_mode_)
        std::cout << "[Schema] Parsing command schema...\n";

    // Basic extraction - find command names and param arrays
    size_t pos = 0;
    while ((pos = json_str.find("\"params\"", pos)) != std::string::npos)
    {
        // Backtrack to find command name
        size_t cmd_start = json_str.rfind("\"", pos - 2);
        if (cmd_start == std::string::npos)
            break;

        // Extract command name
        size_t cmd_name_start = json_str.rfind("\"", cmd_start - 1);
        if (cmd_name_start == std::string::npos)
            break;

        std::string cmd = json_str.substr(cmd_name_start + 1, cmd_start - cmd_name_start - 1);

        CommandSchema schema_entry;
        schema_entry.command = cmd;

        // For now, just store the command name
        // Full param parsing would require more sophisticated JSON handling
        schema_[cmd] = schema_entry;

        if (debug_mode_)
            std::cout << "[Schema] Loaded command: " << cmd << "\n";

        pos++;
    }
}

std::string CommandParamExtractor::get_stats() const
{
    std::ostringstream oss;
    oss << "Embeddings Loaded: " << embeddings_.size() << "\n";
    oss << "Commands in Schema: " << schema_.size() << "\n";

    if (!embeddings_.empty())
    {
        oss << "Embedding Dimension: " << embeddings_.begin()->second.size() << "\n";
    }

    return oss.str();
}

void CommandParamExtractor::dump_embeddings(int max_tokens) const
{
    std::cout << "\n[Embedding Dump] First " << max_tokens << " tokens:\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(30) << "Token"
              << std::setw(40) << "Vector (first 5 dims)"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    int count = 0;
    for (const auto &[token, vec] : embeddings_)
    {
        if (count++ >= max_tokens)
            break;

        std::cout << std::left << std::setw(30) << token;
        std::cout << "[";

        int dims_to_show = std::min(5, static_cast<int>(vec.size()));
        for (int i = 0; i < dims_to_show; i++)
        {
            std::cout << std::fixed << std::setprecision(3) << vec[i];
            if (i < dims_to_show - 1)
                std::cout << ", ";
        }
        if (vec.size() > 5)
            std::cout << ", ...";
        std::cout << "]\n";
    }

    std::cout << std::string(70, '-') << "\n";
    std::cout << "Total embeddings: " << embeddings_.size() << "\n\n";
}
