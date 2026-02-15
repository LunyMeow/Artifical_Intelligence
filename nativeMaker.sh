#!/bin/bash

# Native C++ compilation with all new modules
# Usage: bash nativeMaker.sh [--debug]

DEBUG_FLAG=""
if [ "$1" == "--debug" ]; then
    DEBUG_FLAG="-g -O0"
    echo "[DEBUG MODE] Symbols included, optimization disabled"
else
    DEBUG_FLAG="-O2"
    echo "[RELEASE MODE] Optimized build"
fi

echo "Compiling native binary with parameter extraction and inference engine..."

g++ \
  Buildv1_3_2.cpp \
  ByteBPE/ByteBPETokenizer.cpp \
  CommandParamExtractor.cpp \
  ParameterExtractorV2.cpp \
  InferenceEngine.cpp \
  -o build \
  -std=c++17 \
  $DEBUG_FLAG \
  -Wall -Wextra \
  -I./include \
  -lsqlite3 \
  -pthread

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "   Binary: ./build"
    echo "   Modules compiled:"
    echo "     - Buildv1_3_2.cpp (main neural network)"
    echo "     - ByteBPETokenizer.cpp (BPE tokenization)"
    echo "     - CommandParamExtractor.cpp (parameter extraction)"
    echo "     - InferenceEngine.cpp (unified inference)"
    echo ""
    echo "   Test with:"
    echo "     ./build              (interactive mode)"
    echo "     ./build --wasm-test  (WASM compatibility test)"
    echo ""
else
    echo "❌ Build failed!"
    exit 1
fi
