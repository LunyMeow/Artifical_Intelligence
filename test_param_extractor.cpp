#include "CommandParamExtractor.h"
#include <iostream>
#include <fstream>
#include <sstream>

/**
 * @file test_param_extractor.cpp
 * @brief Debug utility for testing CommandParamExtractor
 * 
 * Usage:
 *   g++ -o test_param_extractor test_param_extractor.cpp CommandParamExtractor.cpp -std=c++17
 *   ./test_param_extractor <embeddings_csv> <command> <sentence>
 * 
 * Example CSV format:
 *   word,dim_0,dim_1,dim_2,...,dim_49
 *   copy,0.12,-0.45,0.23,...
 *   file,0.34,0.56,-0.12,...
 */

// Load embeddings from CSV file
std::unordered_map<std::string, std::vector<float>>
load_embeddings_csv(const std::string &filepath)
{
    std::unordered_map<std::string, std::vector<float>> result;

    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] Cannot open embeddings file: " << filepath << "\n";
        return result;
    }

    std::string line;
    int line_num = 0;

    while (std::getline(file, line))
    {
        line_num++;

        // Skip header or empty lines
        if (line_num == 1 || line.empty())
            continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<float> vec;

        // Read token name
        if (!std::getline(iss, token, ','))
            continue;

        // Trim whitespace from token
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        // Read dimensions
        std::string dim_str;
        while (std::getline(iss, dim_str, ','))
        {
            try
            {
                float val = std::stof(dim_str);
                vec.push_back(val);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[WARNING] Line " << line_num << ": "
                          << "Invalid float value: " << dim_str << "\n";
                break;
            }
        }

        if (vec.size() != CommandParamExtractor::EMBEDDING_DIM)
        {
            std::cerr << "[WARNING] Line " << line_num << ": "
                      << "Expected " << CommandParamExtractor::EMBEDDING_DIM
                      << " dims, got " << vec.size() << " for token: " << token << "\n";
            continue;
        }

        result[token] = vec;
    }

    std::cout << "[INFO] Loaded " << result.size() << " embeddings from " << filepath << "\n";
    return result;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <embeddings_csv> <command> <sentence>\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0]
                  << " embeddings.csv cp \"copy important.txt backup/\"\n";
        return 1;
    }

    std::string embeddings_file = argv[1];
    std::string command = argv[2];
    std::string sentence = argv[3];

    // Load embeddings
    std::cout << "\n[Step 1] Loading embeddings from CSV...\n";
    auto embeddings = load_embeddings_csv(embeddings_file);

    if (embeddings.empty())
    {
        std::cerr << "[ERROR] No embeddings loaded!\n";
        return 1;
    }

    // Create extractor with debug enabled
    std::cout << "\n[Step 2] Initializing CommandParamExtractor...\n";
    CommandParamExtractor extractor(embeddings, "", true); // debug=true

    // Show statistics
    std::cout << "\n[Step 3] Statistics:\n";
    std::cout << extractor.get_stats();

    // Dump first 20 embeddings
    std::cout << "\n[Step 4] Sample embeddings:\n";
    extractor.dump_embeddings(20);

    // Tokenize the sentence
    std::cout << "\n[Step 5] Tokenizing sentence:\n";
    auto tokens = extractor.tokenize(sentence);

    std::cout << "[Result] " << tokens.size() << " tokens extracted:\n";
    for (const auto &tok : tokens)
    {
        std::cout << "  - \"" << tok << "\"\n";
        auto emb = extractor.get_embedding(tok);
        if (emb)
        {
            std::cout << "    ✓ Embedding found (50-dim)\n";
        }
        else
        {
            std::cout << "    ✗ Embedding NOT found (OOV token)\n";
        }
    }

    // Extract parameters
    std::cout << "\n[Step 6] Extracting parameters:\n";
    auto params = extractor.extract(sentence, command);

    std::cout << "[Result] " << params.size() << " parameter(s) extracted:\n";
    for (const auto &[token, score] : params)
    {
        std::cout << "  - \"" << token << "\" (similarity score: " << score << ")\n";
    }

    std::cout << "\n[SUCCESS] Test completed!\n";
    return 0;
}
