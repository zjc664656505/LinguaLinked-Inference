//
// Created by Junchen Zhao on 7/10/23.
//

#ifndef DISTRIBUTED_INFERENCE_DEMO_INFERENCE_H
#define DISTRIBUTED_INFERENCE_DEMO_INFERENCE_H

#include <cstdint>
#include <string>
#include "session_cache.h"
#include <jni.h>
#include "tokenizers_cpp.h"


using tokenizers::Tokenizer;
namespace inference{
    std::vector<Ort::Value> run_inference(SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids);

    std::string LoadBytesFromFile(const std::string& path);

    void PrintEncodeResult(const std::vector<int>& ids);

    std::unique_ptr<Tokenizer> HuggingFaceTokenizer(std::string & tokenizer_file_path);

    std::unique_ptr<Tokenizer> SentencePieceTokenizer(std::string& tokenizer_file_path);

    std::vector<int> Encoding(Tokenizer* tok, std::string & input_string);

    std::string Decoding(Tokenizer* tok, std::vector<int> & output_ids);

    size_t binary_classify(float *logits);

    std::vector<float> Softmax(float *logits, size_t num_logits);

    int GreedyDecoding(std::vector<Ort::Value>& probs);

    double flop_per_second_estimation(int & model_num_flops, SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids);

    std::vector<Ort::Value> extractOrtElementsSingle(std::vector<Ort::Value>& data, std::vector<int>& indices);

    std::vector<std::vector<Ort::Value>> extractOrtElementsGroup(std::vector<Ort::Value>& data, std::vector<std::vector<int>>& indicesGroups);

    std::vector<Ort::Value> combineVectors(const std::vector<Ort::Value>& vec1, const std::vector<Ort::Value>& vec2);

    int run_inference_with_decoding(SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids,
                                    int k,
                                    float initial_temp,
                                    int decoding);

    size_t run_inference_with_binary_classification(SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids, int classification);
}

#endif //DISTRIBUTED_INFERENCE_DEMO_INFERENCE_H
