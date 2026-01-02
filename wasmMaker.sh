#!/bin/bash

emcc Buildv1_3_2.cpp \
  -o model.js \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_load_user_model","_run_inference","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="ModuleFactory" \
  -s INITIAL_MEMORY=33554432 \
  -O3 \
  --no-entry

echo "Build complete: model.js and model.wasm"
