#!/bin/bash

emcc Buildv1_3_2.cpp ByteBPE/ByteBPETokenizer.cpp \
  -o web/public/wasm/model.js \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_load_user_model","_run_inference","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="createModule" \
  -s EXPORT_ES6=1 \
  -s INITIAL_MEMORY=33554432 \
  -s EXCEPTION_CATCHING_ALLOWED="['.*']" \
  -s NO_DISABLE_EXCEPTION_CATCHING=1 \
  -O3 \
  -std=c++17 \
  -I./include

echo "✅ Build complete: web/public/wasm/model.js and model.wasm"
echo "Exception handling: ENABLED"
