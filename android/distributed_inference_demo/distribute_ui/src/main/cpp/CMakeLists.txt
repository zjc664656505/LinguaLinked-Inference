# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.18.1)

project(distributed_inference_demo C CXX)
find_library( log-lib log )

add_library(
        # Sets the name of the library.
        distributed_inference_demo
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        native-lib.cpp
        utils.cpp
        inference.cpp
        decoding.cpp
)

set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/tokenizers-cpp)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers_cpp)


target_include_directories(distributed_inference_demo PRIVATE
        ${CMAKE_SOURCE_DIR}/include/
        ${TOKENIZER_CPP_PATH}/include/
        ${SENTENCE_PIECE_PATH}/include
        )


add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/arm64-v8a/libonnxruntime.so)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
target_link_libraries(
        distributed_inference_demo
        sentencepiece-static
        tokenizers_c
        tokenizers_cpp
        onnxruntime
        log
        ${log-lib}
)
