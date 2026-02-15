#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "InferenceEngine.h"

int main()
{
    std::cout << "\n=== TEST: Parameter Extraction & Template Filling ===\n\n";
    
    // Test 1: Basic fill_template test
    std::vector<std::pair<std::string, float>> params = {
        {"yedekleri", 0.9f},
        {"backup/", 0.8f}
    };
    
    std::cout << "[TEST 1] Testing fill_template() manually\n";
    std::cout << "Command: cp\n";
    std::cout << "Parameters: [yedekleri (0.9), backup/ (0.8)]\n";
    std::cout << "Expected output: cp yedekleri backup/\n\n";
    
    // We would call fill_template here if we had an instance
    // For now just show what we expect
    
    std::cout << "[TEST 2] Parameter extraction from semantic cues\n";
    std::cout << "Sentence: 'backup/ klasörüne yedekleri kopyala'\n";
    std::cout << "Command: cp\n";
    std::cout << "Placeholders: <SRC>, <DST>\n";
    std::cout << "Expected: <SRC>=yedekleri, <DST>=backup/\n";
    std::cout << "Expected final: cp yedekleri backup/\n\n";
    
    // Test both file and directory detection
    std::cout << "[TEST 3] Heuristic rules:\n";
    std::cout << "  - 'yedekleri' (no /) → FILE/SOURCE\n";
    std::cout << "  - 'backup/' (has /) → DIR/DESTINATION\n";
    std::cout << "  - 'chmod' (digits) → PERMS\n";
    std::cout << "  - 'user:group' → OWNER\n\n";
    
    std::cout << "✓ Test plan complete\n";
    std::cout << "  Next: Run actual program with verbose output\n";
    
    return 0;
}
