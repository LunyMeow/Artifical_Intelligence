cmake_minimum_required(VERSION 3.10)
project(ByteBPETokenizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMP0135 policy'sini ayarla (uyarıyı önlemek için)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# nlohmann/json kütüphanesini dahil et
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# Kaynak dosyaları
add_library(bpe_tokenizer
    ByteBPETokenizer.cpp
)

target_link_libraries(bpe_tokenizer PRIVATE nlohmann_json::nlohmann_json)

# Ana program
add_executable(main main.cpp)
target_link_libraries(main PRIVATE bpe_tokenizer nlohmann_json::nlohmann_json)