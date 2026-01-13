#include "ByteBPETokenizer.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>

std::vector<std::string> read_corpus(const std::string &filename)
{
    std::vector<std::string> corpus;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::istringstream iss(line);
        std::string word;
        while (iss >> word)
        {
            corpus.push_back(word);
        }
    }
    if (corpus.empty())
    {
        std::cout << "\n\n[WARNING] corpus is empty.\n\n";
    }

    return corpus;
}

int main(int argc, char *argv[])
{
    // Corpus'u oku
    if (argc < 2)
    {
        std::cerr << "Kullanim: " << argv[0] << " <corpus_dosya_yolu>\n";
        return 1;
    }

    std::string corpus_path = argv[1];

    // Corpus'u oku
    std::vector<std::string> corpus = read_corpus(corpus_path);
    // Tokenizer oluştur
    ByteBPETokenizer tokenizer(500);

    // Training
    auto t0 = std::chrono::high_resolution_clock::now();
    int vocab_size = tokenizer.train(corpus);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t1 - t0;

    std::cout << "Training vocab size: " << vocab_size << std::endl;
    std::cout << "Training time: " << diff.count() << "s" << std::endl;

    // Test örnekleri
    std::vector<std::string> test_texts = {"dizini", "dizine", "ghkgkfgyı"};

    for (const auto &text : test_texts)
    {
        std::cout << "----------" << std::endl;

        auto ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);

        std::cout << "Input  : " << text << std::endl;
        std::cout << "IDs    : [";
        for (size_t i = 0; i < ids.size(); ++i)
        {
            std::cout << ids[i];
            if (i < ids.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Decoded: " << decoded << std::endl;
    }

    // Model kaydet
    tokenizer.save("tokenizer_model.json");
    std::cout << "\nModel saved to tokenizer_model.json" << std::endl;

    // Model yükle (örnek)
    ByteBPETokenizer loaded_tokenizer;
    loaded_tokenizer.load("tokenizer_model.json");
    std::cout << "Model loaded successfully!" << std::endl;

    return 0;
}