#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <future>
#include <thread>




using namespace std;

// Thread-safe logging system
class Logger {
private:
    static mutex log_mutex;
    static bool debug_mode;
    static string log_file;
    
public:
    static void setDebugMode(bool mode) { debug_mode = mode; }
    static void setLogFile(const string& filename) { log_file = filename; }
    
    static void log(const string& message, const string& level = "INFO") {
        if (!debug_mode && level == "DEBUG") return;
        
        lock_guard<mutex> lock(log_mutex);
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        
        stringstream ss;
        ss << "[" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
           << "[" << level << "] " << message;
        
        cout << ss.str() << endl;
        
        if (!log_file.empty()) {
            ofstream out(log_file, ios::app);
            if (out.is_open()) {
                out << ss.str() << endl;
                out.close();
            }
        }
    }
};

mutex Logger::log_mutex;
bool Logger::debug_mode = true;
string Logger::log_file = "llm_training.log";

// Tokenizer for text processing

enum class TokenizerMode {
    WORD,        // mevcut sistem
    CHAR_NGRAM   // harf parçalama
};


class Tokenizer {
private:
    unordered_map<string, int> tokenToId;
    unordered_map<int, string> idToToken;
    vector<string> vocabulary;
    int vocabSize;

    TokenizerMode mode;
    int charBlockSize;

    const string EOS_TOKEN = "<EOS>";
    const string UNK_TOKEN = "<UNK>";

    // Temel normalize fonksiyonu
    string normalize(string word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end());
        return word;
    }

    // Kelimeyi N harflik parçalara böl
    vector<string> splitToCharBlocks(const string& word) {
        vector<string> blocks;
        for (size_t i = 0; i < word.size(); i += charBlockSize) {
            blocks.push_back(word.substr(i, charBlockSize));
        }
        return blocks;
    }

public:
    Tokenizer(
        TokenizerMode mode = TokenizerMode::WORD,
        int charBlockSize = 3
    )
        : vocabSize(0), mode(mode), charBlockSize(charBlockSize) {}

    void buildVocabulary(const vector<string>& texts) {
        unordered_set<string> uniqueTokens;

        uniqueTokens.insert(EOS_TOKEN);
        uniqueTokens.insert(UNK_TOKEN);

        for (const auto& text : texts) {
            stringstream ss(text);
            string word;

            while (ss >> word) {
                word = normalize(word);
                if (word.empty()) continue;

                if (mode == TokenizerMode::WORD) {
                    uniqueTokens.insert(word);
                } else {
                    auto blocks = splitToCharBlocks(word);
                    for (const auto& b : blocks)
                        uniqueTokens.insert(b);
                }
            }
        }

        vocabulary.assign(uniqueTokens.begin(), uniqueTokens.end());
        vocabSize = vocabulary.size();

        for (int i = 0; i < vocabSize; i++) {
            tokenToId[vocabulary[i]] = i;
            idToToken[i] = vocabulary[i];
        }

        Logger::log(
            "Tokenizer vocabulary built | Size: " +
            to_string(vocabSize) +
            " | Mode: " +
            (mode == TokenizerMode::WORD ? "WORD" : "CHAR_NGRAM")
        );
    }

    vector<int> encode(const string& text) {
        vector<int> tokens;
        stringstream ss(text);
        string word;

        while (ss >> word) {
            word = normalize(word);
            if (word.empty()) continue;

            if (mode == TokenizerMode::WORD) {
                auto it = tokenToId.find(word);
                tokens.push_back(it != tokenToId.end()
                                     ? it->second
                                     : tokenToId[UNK_TOKEN]);
            } else {
                auto blocks = splitToCharBlocks(word);
                for (const auto& b : blocks) {
                    auto it = tokenToId.find(b);
                    tokens.push_back(it != tokenToId.end()
                                         ? it->second
                                         : tokenToId[UNK_TOKEN]);
                }
            }
        }

        // Cümle sonu token
        tokens.push_back(tokenToId[EOS_TOKEN]);

        return tokens;
    }

    string decode(const vector<int>& tokens) {
        string text;

        for (int token : tokens) {
            if (token >= 0 && token < vocabSize) {
                const string& t = idToToken[token];
                if (t == EOS_TOKEN) {
                    text += " <EOS>";
                } else {
                    text += t + " ";
                }
            } else {
                text += UNK_TOKEN + " ";
            }
        }

        if (!text.empty() && text.back() == ' ')
            text.pop_back();

        return text;
    }

    int getVocabSize() const { return vocabSize; }
};


// Embedding Layer
class EmbeddingLayer {
private:
    int vocabSize;
    int embeddingDim;
    vector<vector<double>> embeddingMatrix;
    double learningRate;
    mt19937 rng;
    
public:
    EmbeddingLayer(int vocab, int embedDim, double lr = 0.001) 
        : vocabSize(vocab), embeddingDim(embedDim), learningRate(lr), 
          rng(chrono::steady_clock::now().time_since_epoch().count()) {
        
        embeddingMatrix.resize(vocabSize, vector<double>(embeddingDim));
        initializeEmbeddings();
    }
    
    void initializeEmbeddings() {
        uniform_real_distribution<double> dist(-0.1, 0.1);
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddingMatrix[i][j] = dist(rng);
            }
        }
        
        Logger::log("Initialized embedding matrix: " + to_string(vocabSize) + "x" + to_string(embeddingDim));
    }
    
    vector<double> getEmbedding(int tokenId) const {
        if (tokenId < 0 || tokenId >= vocabSize) {
            return vector<double>(embeddingDim, 0.0);
        }
        return embeddingMatrix[tokenId];
    }
    
    vector<vector<double>> getEmbeddings(const vector<int>& tokenIds) const {
        vector<vector<double>> embeddings;
        embeddings.reserve(tokenIds.size());
        
        for (int tokenId : tokenIds) {
            embeddings.push_back(getEmbedding(tokenId));
        }
        
        return embeddings;
    }
    
    void updateEmbedding(int tokenId, const vector<double>& gradient) {
        if (tokenId >= 0 && tokenId < vocabSize && static_cast<int>(gradient.size()) == embeddingDim) {
            for (int i = 0; i < embeddingDim; i++) {
                embeddingMatrix[tokenId][i] -= learningRate * gradient[i];
            }
        }
    }
    
    void setLearningRate(double lr) { learningRate = lr; }
    int getEmbeddingDim() const { return embeddingDim; }
};

// Positional Encoding
class PositionalEncoding {
private:
    int maxSequenceLength;
    int embeddingDim;
    vector<vector<double>> encodings;
    
public:
    PositionalEncoding(int maxLen, int embedDim) 
        : maxSequenceLength(maxLen), embeddingDim(embedDim) {
        generateEncodings();
    }
    
    void generateEncodings() {
        encodings.resize(maxSequenceLength, vector<double>(embeddingDim, 0.0));
        
        for (int pos = 0; pos < maxSequenceLength; pos++) {
            for (int i = 0; i < embeddingDim; i++) {
                if (i % 2 == 0) {
                    encodings[pos][i] = sin(pos / pow(10000.0, (2.0 * i) / embeddingDim));
                } else {
                    encodings[pos][i] = cos(pos / pow(10000.0, (2.0 * (i-1)) / embeddingDim));
                }
            }
        }
        
        Logger::log("Generated positional encodings for max length: " + to_string(maxSequenceLength));
    }
    
    vector<vector<double>> apply(const vector<vector<double>>& embeddings) const {
        vector<vector<double>> result = embeddings;
        int seqLength = min((int)embeddings.size(), maxSequenceLength);
        
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < embeddingDim && j < static_cast<int>(embeddings[i].size()); j++) {
                result[i][j] += encodings[i][j];
            }
        }
        
        return result;
    }
};

// Multi-Head Attention
class MultiHeadAttention {
private:
    int embedDim;
    int numHeads;
    int headDim;
    vector<vector<vector<double>>> queryWeights;  // [head][embed][headDim]
    vector<vector<vector<double>>> keyWeights;
    vector<vector<vector<double>>> valueWeights;
    vector<vector<double>> outputWeights;        // [embed][embed]
    vector<double> outputBias;
    double learningRate;
    mt19937 rng;
    
    vector<double> softmax(const vector<double>& scores) const {
        vector<double> result(scores.size());
        double maxScore = *max_element(scores.begin(), scores.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < scores.size(); i++) {
            result[i] = exp(scores[i] - maxScore);
            sum += result[i];
        }
        
        for (size_t i = 0; i < result.size(); i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    double dotProduct(const vector<double>& a, const vector<double>& b) const {
        double result = 0.0;
        for (size_t i = 0; i < min(a.size(), b.size()); i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
public:
    MultiHeadAttention(int embedDim, int numHeads, double lr = 0.001) 
        : embedDim(embedDim), numHeads(numHeads), learningRate(lr),
          rng(chrono::steady_clock::now().time_since_epoch().count()) {
        
        if (embedDim % numHeads != 0) {
            throw invalid_argument("embedDim must be divisible by numHeads");
        }
        
        headDim = embedDim / numHeads;
        initializeWeights();
    }
    
    void initializeWeights() {
        uniform_real_distribution<double> dist(-0.1, 0.1);
        
        // Initialize Q, K, V weights for each head
        queryWeights.resize(numHeads, vector<vector<double>>(embedDim, vector<double>(headDim)));
        keyWeights.resize(numHeads, vector<vector<double>>(embedDim, vector<double>(headDim)));
        valueWeights.resize(numHeads, vector<vector<double>>(embedDim, vector<double>(headDim)));
        
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < embedDim; i++) {
                for (int j = 0; j < headDim; j++) {
                    queryWeights[h][i][j] = dist(rng);
                    keyWeights[h][i][j] = dist(rng);
                    valueWeights[h][i][j] = dist(rng);
                }
            }
        }
        
        // Initialize output projection weights
        outputWeights.resize(embedDim, vector<double>(embedDim));
        outputBias.resize(embedDim);
        
        for (int i = 0; i < embedDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                outputWeights[i][j] = dist(rng);
            }
            outputBias[i] = dist(rng);
        }
        
        Logger::log("Initialized multi-head attention: " + to_string(numHeads) + " heads, " + to_string(embedDim) + " dimensions");
    }
    
    vector<vector<double>> forward(const vector<vector<double>>& input) const {
        int seqLength = input.size();
        
        // Project to Q, K, V for each head
        vector<vector<vector<double>>> queries(numHeads);
        vector<vector<vector<double>>> keys(numHeads);
        vector<vector<vector<double>>> values(numHeads);
        
        for (int h = 0; h < numHeads; h++) {
            queries[h].resize(seqLength, vector<double>(headDim));
            keys[h].resize(seqLength, vector<double>(headDim));
            values[h].resize(seqLength, vector<double>(headDim));
            
            for (int i = 0; i < seqLength; i++) {
                for (int j = 0; j < embedDim; j++) {
                    for (int d = 0; d < headDim; d++) {
                        queries[h][i][d] += input[i][j] * queryWeights[h][j][d];
                        keys[h][i][d] += input[i][j] * keyWeights[h][j][d];
                        values[h][i][d] += input[i][j] * valueWeights[h][j][d];
                    }
                }
            }
        }
        
        // Compute attention for each head
        vector<vector<vector<double>>> headOutputs(numHeads, vector<vector<double>>(seqLength, vector<double>(headDim)));
        
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLength; i++) {
                vector<double> attentionScores(seqLength);
                
                // Compute attention scores
                for (int j = 0; j < seqLength; j++) {
                    double score = dotProduct(queries[h][i], keys[h][j]) / sqrt(headDim);
                    attentionScores[j] = score;
                }
                
                // Apply softmax
                vector<double> attentionWeights = softmax(attentionScores);
                
                // Apply attention to values
                for (int d = 0; d < headDim; d++) {
                    headOutputs[h][i][d] = 0.0;
                    for (int j = 0; j < seqLength; j++) {
                        headOutputs[h][i][d] += attentionWeights[j] * values[h][j][d];
                    }
                }
            }
        }
        
        // Concatenate heads and project output
        vector<vector<double>> output(seqLength, vector<double>(embedDim, 0.0));
        
        for (int i = 0; i < seqLength; i++) {
            vector<double> concatenated(embedDim);
            
            // Concatenate all heads
            for (int h = 0; h < numHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    concatenated[h * headDim + d] = headOutputs[h][i][d];
                }
            }
            
            // Project to output
            for (int j = 0; j < embedDim; j++) {
                output[i][j] = outputBias[j];
                for (int k = 0; k < embedDim; k++) {
                    output[i][j] += concatenated[k] * outputWeights[k][j];
                }
            }
        }
        
        return output;
    }
    
    int getNumHeads() const { return numHeads; }
    int getEmbedDim() const { return embedDim; }
};

// Enhanced CorticalColumn with LLM capabilities
class CorticalColumn {
private:
    unique_ptr<Tokenizer> tokenizer;
    unique_ptr<EmbeddingLayer> embeddingLayer;
    unique_ptr<PositionalEncoding> positionalEncoding;
    vector<unique_ptr<MultiHeadAttention>> attentionLayers;
    
    string activationType;
    double learningRate;
    bool debug;
    
    // Performance monitoring
    vector<double> trainingLosses;
    vector<double> perplexityScores;
    vector<double> layerLosses;
    
    // Training parameters
    int maxSequenceLength;
    int vocabSize;
    int embeddingDim;
    int numLayers;
    int numHeads;
    
public:
    CorticalColumn(const string& activation = "tanh") 
        : activationType(activation), learningRate(0.001), debug(true),
          maxSequenceLength(512), vocabSize(10000), embeddingDim(512), 
          numLayers(6), numHeads(8) {
        
        Logger::setDebugMode(debug);
        Logger::log("Initializing LLM Cortical Column");
        
        try {
            tokenizer = make_unique<Tokenizer>();
            embeddingLayer = make_unique<EmbeddingLayer>(vocabSize, embeddingDim, learningRate);
            positionalEncoding = make_unique<PositionalEncoding>(maxSequenceLength, embeddingDim);
            
            // Initialize attention layers
            for (int i = 0; i < numLayers; i++) {
                attentionLayers.push_back(
                    make_unique<MultiHeadAttention>(embeddingDim, numHeads, learningRate)
                );
            }
            
            Logger::log("LLM Cortical Column initialized successfully");
            
        } catch (const exception& e) {
            Logger::log("Failed to initialize LLM: " + string(e.what()), "ERROR");
            throw;
        }
    }
    
    // Initialize LLM with custom parameters
    void initializeLLM(int vocab, int embedDim, int layers, int heads, int maxSeqLen = 512) {
        vocabSize = vocab;
        embeddingDim = embedDim;
        numLayers = layers;
        numHeads = heads;
        maxSequenceLength = maxSeqLen;
        
        if (embedDim % heads != 0) {
            throw invalid_argument("embeddingDim must be divisible by numHeads");
        }
        
        Logger::log("Initializing custom LLM: vocab=" + to_string(vocab) + 
                   ", embed=" + to_string(embedDim) + ", layers=" + to_string(layers) + 
                   ", heads=" + to_string(heads));
        
        tokenizer = make_unique<Tokenizer>();
        embeddingLayer = make_unique<EmbeddingLayer>(vocabSize, embeddingDim, learningRate);
        positionalEncoding = make_unique<PositionalEncoding>(maxSequenceLength, embeddingDim);
        
        attentionLayers.clear();
        for (int i = 0; i < numLayers; i++) {
            attentionLayers.push_back(
                make_unique<MultiHeadAttention>(embeddingDim, numHeads, learningRate)
            );
        }
    }
    
    // Build vocabulary from training texts
    void buildVocabulary(const vector<string>& trainingTexts) {
        if (!tokenizer) {
            throw runtime_error("Tokenizer not initialized");
        }
        
        tokenizer->buildVocabulary(trainingTexts);
        vocabSize = tokenizer->getVocabSize();
        
        Logger::log("Vocabulary built with " + to_string(vocabSize) + " tokens");
    }
    
    // Forward pass for text
    vector<vector<double>> forward(const string& inputText) {
        if (!tokenizer || !embeddingLayer || !positionalEncoding) {
            throw runtime_error("LLM components not properly initialized");
        }
        
        // Tokenize input
        vector<int> tokens = tokenizer->encode(inputText);
        
        // Handle sequences longer than max length
        if (tokens.size() > static_cast<size_t>(maxSequenceLength)) {
            tokens.resize(maxSequenceLength);
        }
        
        // Get embeddings
        vector<vector<double>> embeddings = embeddingLayer->getEmbeddings(tokens);
        
        // Apply positional encoding
        embeddings = positionalEncoding->apply(embeddings);
        
        // Pass through attention layers
        vector<vector<double>> output = embeddings;
        for (auto& attentionLayer : attentionLayers) {
            output = attentionLayer->forward(output);
        }
        
        return output;
    }
    
    // Training on text
    double trainOnText(const string& inputText, const string& targetText) {
        try {
            // Forward pass
            auto inputOutput = forward(inputText);
            auto targetOutput = forward(targetText);
            
            // Compute loss (simplified cross-entropy)
            double loss = computeCrossEntropyLoss(inputOutput, targetOutput);
            
            // Store loss for monitoring
            trainingLosses.push_back(loss);
            if (trainingLosses.size() > 1000) {
                trainingLosses.erase(trainingLosses.begin());
            }
            
            // Compute perplexity
            double perplexity = exp(loss);
            perplexityScores.push_back(perplexity);
            if (perplexityScores.size() > 100) {
                perplexityScores.erase(perplexityScores.begin());
            }
            
            // Monitor and adapt (using existing dynamic system)
            monitorLLMPerformance();
            
            Logger::log("Training step - Loss: " + to_string(loss) + 
                       ", Perplexity: " + to_string(perplexity));
            
            return loss;
            
        } catch (const exception& e) {
            Logger::log("Training error: " + string(e.what()), "ERROR");
            return -1.0;
        }
    }
    
    // Monitor LLM performance and adapt dynamically
    int monitorLLMPerformance() {
        if (trainingLosses.size() < 10) return 0;
        
        double slope = computeErrorSlope(50);
        double currentLoss = trainingLosses.back();
        double currentPerplexity = perplexityScores.empty() ? 0 : perplexityScores.back();
        double targetPerplexity = 10.0; // Target perplexity
        
        Logger::log("Performance monitoring - Slope: " + to_string(slope) + 
                   ", Loss: " + to_string(currentLoss) + 
                   ", Perplexity: " + to_string(currentPerplexity));
        
        // Adaptive architecture changes based on performance
        if (abs(slope) < 0.001 && currentPerplexity > targetPerplexity) {
            // Try to improve architecture
            if (currentPerplexity > targetPerplexity * 2) {
                // Significant performance issues - try adding attention heads
                if (addAttentionHead()) {
                    Logger::log("Added attention head to improve performance", "INFO");
                    return 4; // Architecture changed
                }
            }
            
            // Try increasing embedding dimension
            if (currentPerplexity > targetPerplexity * 1.5) {
                if (expandEmbeddingDimension()) {
                    Logger::log("Expanded embedding dimension to improve performance", "INFO");
                    return 4; // Architecture changed
                }
            }
        }
        
        // Learning rate adaptation
        if (slope > 0.01) {
            // Loss increasing - reduce learning rate
            learningRate *= 0.9;
            updateLearningRates();
            Logger::log("Reduced learning rate to: " + to_string(learningRate), "DEBUG");
            return 1; // Learning rate changed
        } else if (abs(slope) < 0.001 && currentLoss > 5.0) {
            // Stuck in local minimum - increase learning rate
            learningRate *= 1.1;
            updateLearningRates();
            Logger::log("Increased learning rate to: " + to_string(learningRate), "DEBUG");
            return 1; // Learning rate changed
        }
        
        return 0; // No changes
    }
    
    // Add attention head dynamically
    bool addAttentionHead() {
        if (numHeads >= 16 || embeddingDim % (numHeads + 1) != 0) {
            return false; // Cannot add more heads
        }
        
        try {
            numHeads++;
            
            // Reinitialize attention layers with new head count
            for (auto& attentionLayer : attentionLayers) {
                attentionLayer = make_unique<MultiHeadAttention>(embeddingDim, numHeads, learningRate);
            }
            
            Logger::log("Successfully added attention head. Total heads: " + to_string(numHeads));
            return true;
            
        } catch (const exception& e) {
            Logger::log("Failed to add attention head: " + string(e.what()), "ERROR");
            return false;
        }
    }
    
    // Expand embedding dimension
    bool expandEmbeddingDimension() {
        int newDim = embeddingDim * 1.2;
        
        if (newDim > 1024 || newDim % numHeads != 0) {
            return false; // Cannot expand further
        }
        
        try {
            // Reinitialize with new embedding dimension
            embeddingDim = newDim;
            
            embeddingLayer = make_unique<EmbeddingLayer>(vocabSize, embeddingDim, learningRate);
            positionalEncoding = make_unique<PositionalEncoding>(maxSequenceLength, embeddingDim);
            
            for (auto& attentionLayer : attentionLayers) {
                attentionLayer = make_unique<MultiHeadAttention>(embeddingDim, numHeads, learningRate);
            }
            
            Logger::log("Successfully expanded embedding dimension to: " + to_string(embeddingDim));
            return true;
            
        } catch (const exception& e) {
            Logger::log("Failed to expand embedding dimension: " + string(e.what()), "ERROR");
            return false;
        }
    }
    
    // Compute cross-entropy loss
    double computeCrossEntropyLoss(const vector<vector<double>>& logits, 
                                  const vector<vector<double>>& targets) {
        double loss = 0.0;
        int sequenceLength = min(logits.size(), targets.size());
        
        for (int i = 0; i < sequenceLength; i++) {
            // Simplified loss computation
            double logProb = 0.0;
            double sumExp = 0.0;
            
            // Compute softmax
            for (double logit : logits[i]) {
                sumExp += exp(logit);
            }
            
            // Use first element of target as target index (simplified)
            if (!targets[i].empty()) {
                int targetIndex = (int)targets[i][0] % logits[i].size();
                if (targetIndex >= 0 && targetIndex < static_cast<int>(logits[i].size())) {
                    logProb = log(exp(logits[i][targetIndex]) / sumExp);
                }
            }
            
            loss -= logProb;
        }
        
        return loss / max(sequenceLength, 1);
    }
    
    // Compute error slope for monitoring
    double computeErrorSlope(int windowSize = 50) const {
        if (trainingLosses.size() < 2) return 0.0;
        
        int startIdx = max(0, (int)trainingLosses.size() - windowSize);
        int endIdx = trainingLosses.size() - 1;
        
        if (endIdx <= startIdx) return 0.0;
        
        // Simple linear regression slope
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = endIdx - startIdx + 1;
        
        for (int i = startIdx; i <= endIdx; i++) {
            double x = i - startIdx;
            double y = trainingLosses[i];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return slope;
    }
    
    // Update learning rates across all components
    void updateLearningRates() {
        if (embeddingLayer) {
            embeddingLayer->setLearningRate(learningRate);
        }
        
        // MultiHeadAttention learning rate updates would go here
        // for (auto& attentionLayer : attentionLayers) {
        //     attentionLayer->setLearningRate(learningRate);
        // }
    }
    
    // Get current statistics
    struct LLMStats {
        double currentLoss;
        double currentPerplexity;
        double errorSlope;
        int vocabSize;
        int embeddingDim;
        int numLayers;
        int numHeads;
        double learningRate;
    };
    
    LLMStats getStatistics() const {
        LLMStats stats;
        stats.currentLoss = trainingLosses.empty() ? 0 : trainingLosses.back();
        stats.currentPerplexity = perplexityScores.empty() ? 0 : perplexityScores.back();
        stats.errorSlope = computeErrorSlope();
        stats.vocabSize = vocabSize;
        stats.embeddingDim = embeddingDim;
        stats.numLayers = numLayers;
        stats.numHeads = numHeads;
        stats.learningRate = learningRate;
        
        return stats;
    }
    
    // Save model state
    void saveModel(const string& filename) const {
        ofstream out(filename);
        if (!out.is_open()) {
            throw runtime_error("Cannot open file for saving: " + filename);
        }
        
        // Save basic parameters
        out << "vocabSize:" << vocabSize << "\n";
        out << "embeddingDim:" << embeddingDim << "\n";
        out << "numLayers:" << numLayers << "\n";
        out << "numHeads:" << numHeads << "\n";
        out << "maxSequenceLength:" << maxSequenceLength << "\n";
        out << "learningRate:" << learningRate << "\n";
        out << "activationType:" << activationType << "\n";
        
        // Save training statistics
        out << "trainingLosses:";
        for (double loss : trainingLosses) {
            out << loss << ",";
        }
        out << "\n";
        
        out.close();
        Logger::log("Model saved to: " + filename);
    }
    
    // Generate text (simplified)
    string generateText(const string& prompt, int maxTokens = 50) {
        if (!tokenizer) {
            return "Tokenizer not initialized";
        }
        
        string result = prompt;
        string currentText = prompt;
        
        for (int i = 0; i < maxTokens; i++) {
            try {
                auto output = forward(currentText);
                
                // Simplified token selection (would need proper softmax sampling)
                int nextToken = rand() % tokenizer->getVocabSize();
                string nextWord = tokenizer->decode({nextToken});
                
                result += " " + nextWord;
                currentText += " " + nextWord;
                
            } catch (const exception& e) {
                Logger::log("Generation error: " + string(e.what()), "ERROR");
                break;
            }
        }
        
        return result;
    }
    
    // Setters
    void setDebug(bool mode) { 
        debug = mode; 
        Logger::setDebugMode(mode);
    }
    
    void setLearningRate(double lr) { 
        learningRate = lr; 
        updateLearningRates();
    }
    
    void setActivationType(const string& type) { activationType = type; }
};

// Testing utilities
class LLMTester {
public:
    static void runBasicTests() {
        Logger::log("Running basic LLM tests...");
        
        try {
            // Test tokenizer
            testTokenizer();
            
            // Test embedding layer
            testEmbeddingLayer();
            
            // Test attention mechanism
            testAttention();
            
            // Test full LLM
            testFullLLM();
            
            Logger::log("All tests passed successfully!");
            
        } catch (const exception& e) {
            Logger::log("Test failed: " + string(e.what()), "ERROR");
            throw;
        }
    }
    
private:
    static void testTokenizer() {
        Tokenizer tokenizer;
        vector<string> texts = {"hello world", "test sentence", "another example"};
        
        tokenizer.buildVocabulary(texts);
        
        auto tokens = tokenizer.encode("hello world");
        string decoded = tokenizer.decode(tokens);
        
        if (decoded != "hello world") {
            throw runtime_error("Tokenizer test failed");
        }
        
        Logger::log("Tokenizer test passed");
    }
    
    static void testEmbeddingLayer() {
        EmbeddingLayer embedding(100, 64);
        
        auto embeddingVector = embedding.getEmbedding(5);
        if (embeddingVector.size() != 64) {
            throw runtime_error("Embedding layer dimension test failed");
        }
        
        Logger::log("Embedding layer test passed");
    }
    
    static void testAttention() {
        MultiHeadAttention attention(64, 4);
        
        vector<vector<double>> input(10, vector<double>(64, 0.1));
        auto output = attention.forward(input);
        
        if (output.size() != 10 || output[0].size() != 64) {
            throw runtime_error("Attention mechanism test failed");
        }
        
        Logger::log("Attention mechanism test passed");
    }
    
    static void testFullLLM() {
        CorticalColumn llm;
        
        vector<string> trainingTexts = {
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological neural networks"
        };
        
        llm.buildVocabulary(trainingTexts);
        
        string testText = "The quick brown";
        auto output = llm.forward(testText);
        
        if (output.empty()) {
            throw runtime_error("LLM forward pass test failed");
        }
        
        // Test training
        double loss = llm.trainOnText(testText, testText);
        if (loss < 0) {
            throw runtime_error("LLM training test failed");
        }
        
        Logger::log("Full LLM test passed");
    }
};

int main() {
    try {
        Logger::log("Starting LLM Interactive Mode");

        CorticalColumn llm("tanh");
        llm.initializeLLM(1000, 256, 4, 4, 128);

        vector<string> trainingTexts = {
            "Hava bugün çok güzel",
            "Hava bugün sıcak",
            "Okula gidiyorum"
        };

        llm.buildVocabulary(trainingTexts);

        // Basit ön eğitim
        for (int epoch = 0; epoch < 5; epoch++) {
            for (const string& text : trainingTexts) {
                llm.trainOnText(text, text);
            }
        }

        Logger::log("Interactive chat started (type 'exit' to quit)");

        while (true) {
            std::cout << "\n> ";
            std::string input;
            std::getline(std::cin, input);

            if (input == "exit" || input == "quit")
                break;

            std::string prompt = input;
            int tokenCount = 20; // default

            // "-" kontrolü
            size_t dashPos = input.find('-');
            if (dashPos != std::string::npos) {
                prompt = input.substr(0, dashPos);
                std::string numberPart = input.substr(dashPos + 1);

                try {
                    tokenCount = std::stoi(numberPart);
                } catch (...) {
                    Logger::log("Invalid token count, using default (20)", "WARN");
                    tokenCount = 20;
                }
            }

            // boşluk temizleme
            while (!prompt.empty() && prompt.back() == ' ')
                prompt.pop_back();

            std::string response = llm.generateText(prompt, tokenCount);

            std::cout << "LLM: " << response << std::endl;
        }

        llm.saveModel("llm_model.txt");
        Logger::log("Model saved. Exiting.");

    } catch (const std::exception& e) {
        Logger::log("Fatal error: " + std::string(e.what()), "ERROR");
        return 1;
    }

    return 0;
}
