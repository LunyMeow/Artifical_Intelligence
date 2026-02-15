#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "InferenceEngine.h"

using namespace std;

int main()
{
    cout << "\n=== InferenceEngine Integration Test ===\n\n";
    
    // Minimal embeddings setup for testing
    unordered_map<string, vector<float>> word_embeddings;
    unordered_map<string, vector<float>> cmd_embeddings;
    
    // Add sample embeddings
    vector<float> emb1(50, 0.1f);
    vector<float> emb2(50, 0.2f);
    vector<float> emb3(50, 0.3f);
    vector<float> emb4(50, 0.4f);
    vector<float> emb5(50, 0.5f);
    
    // Better embeddings for the test
    word_embeddings["backup"] = emb1;
    word_embeddings["klasorune"] = emb2;  // destination indicator
    word_embeddings["yedekleri"] = emb3;  // source file
    word_embeddings["kopyala"] = emb4;
    word_embeddings["terminali"] = emb1;
    word_embeddings["ac"] = emb2;
    word_embeddings["klasor"] = emb2;     // directory
    word_embeddings["dosya"] = emb3;      // file
    
    cmd_embeddings["cp"] = emb1;
    cmd_embeddings["mkdir"] = emb2;
    cmd_embeddings["rm"] = emb3;
    
    // Create parameter extractor (without embeddings for params)
    CommandParamExtractor* extractor = new CommandParamExtractor(
        word_embeddings, "", true);
    
    // Create inference engine  
    InferenceEngine engine(
        extractor,
        word_embeddings,
        cmd_embeddings,
        true); // debug mode
    
    cout << "[TEST 1] Testing extract_parameters_from_sentence() for CP command\n";
    cout << "Sentence: 'backup klasorune yedekleri kopyala'\n";
    cout << "Expected: <SRC>=yedekleri, <DST>=backup\n";
    cout << "Placeholders: <SRC>, <DST>\n\n";
    
    vector<string> placeholders = {"<SRC>", "<DST>"};
    auto params = engine.extract_parameters_from_sentence(
        "backup klasorune yedekleri kopyala",
        placeholders
    );
    
    cout << "\n[RESULT] Extracted parameters:\n";
    for (const auto& [param, score] : params) {
        cout << "  " << param << " (confidence: " << score << ")\n";
    }
    
    cout << "\n[TEST 2] Testing fill_template()\n";
    string filled = engine.fill_template("cp", params);
    cout << "[RESULT] Filled template: " << filled << "\n";
    
    cout << "\n[TEST 3] Testing with MKDIR command\n";
    vector<string> mkdir_placeholders = {"<DIR>"};
    auto mkdir_params = engine.extract_parameters_from_sentence(
        "backup klasor olustur",
        mkdir_placeholders
    );
    
    cout << "[RESULT] Extracted parameters for mkdir:\n";
    for (const auto& [param, score] : mkdir_params) {
        cout << "  " << param << " (confidence: " << score << ")\n";
    }
    
    string mkdir_filled = engine.fill_template("mkdir", mkdir_params);
    cout << "[RESULT] Filled mkdir template: " << mkdir_filled << "\n";
    
    delete extractor;
    
    cout << "\n[TEST 4] Testing JSON schema loading\n";
    bool loaded = engine.init_with_schema("LLM/Embeddings/cmdparam/command_schema.json");
    
    if (loaded)
    {
        cout << "[SUCCESS] Command schema loaded from JSON!\n";
    }
    else
    {
        cout << "[INFO] Schema file not found, using fallback templates\n";
    }
    
    cout << "\n✓ All tests completed\n";
    return 0;
}
