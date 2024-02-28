//
// Created by Junchen Zhao on 7/10/23.
//
#include "inference.h"
#include <cassert>
#include "onnxruntime_cxx_api.h"
#include <string>
#include <iostream>
#include <jni.h>
#include "tokenizers-cpp/include/tokenizers_cpp.h"
#include <fstream>
#include <chrono>
#include "utils.h"
#include "android/log.h"
#include <numeric>
#include <random>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "decoding.h"
using tokenizers::Tokenizer;

namespace inference{
    std::string LoadBytesFromFile(const std::string& path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    std::vector<float> Softmax(float *logits, size_t num_logits) {
        std::vector<float> probabilities(num_logits, 0);
        float sum = 0;
        for (size_t i = 0; i < num_logits; ++i) {
            probabilities[i] = exp(logits[i]);
            sum += probabilities[i];
        }

        if (sum != 0.0f) {
            for (size_t i = 0; i < num_logits; ++i) {
                probabilities[i] /= sum;
            }
        }

        return probabilities;
    }

    size_t binary_classify(float *logits) {
        size_t num_logits = 2;
        size_t idx_max = 0;
        float max_val = logits[0];

        for (size_t i = 1; i < num_logits; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                idx_max = i;
            }
        }
        return idx_max;
    }

    void PrintEncodeResult(const std::vector<int>& ids) {
        std::cout << "tokens=[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]" << std::endl;
    }

    std::unique_ptr<Tokenizer> HuggingFaceTokenizer(std::string & tokenizer_file_path) {
        auto blob = LoadBytesFromFile(tokenizer_file_path);
        // Note: all the current factory APIs takes in-memory blob as input.
        // This gives some flexibility on how these blobs can be read.
        auto tok = Tokenizer::FromBlobJSON(blob);
        return std::move(tok);
    }

    std::unique_ptr<Tokenizer> SentencePieceTokenizer(std::string& tokenizer_file_path){
        auto blob = LoadBytesFromFile(tokenizer_file_path);
        // Note: all the current factory APIs takes in-memory blob as input.
        // This gives some flexibility on how these blobs can be read.
        auto tok = Tokenizer::FromBlobSentencePiece(blob);
        return std::move(tok);
    }

    std::vector<int> Encoding(Tokenizer* tok, std::string & input_string) {
        std::string prompt = input_string;
        std::vector<int> ids = tok->Encode(prompt);
        return ids;
    }

    std::string Decoding(Tokenizer* tok, std::vector<int> & output_ids) {
        std::string decoded_prompt = tok->Decode(output_ids);
        return decoded_prompt;
    }

    int GreedyDecoding(std::vector<Ort::Value>& probs) {
        // TOP-k Sampling
        auto probs_shape = probs.front().GetTensorTypeAndShapeInfo().GetShape();
        int dim2 = probs_shape[1];  // sequence length
        int dim3 = probs_shape[2];  // vocabulary size
        float* tensor_prob = probs.front().GetTensorMutableData<float>();
        int start_index = (dim2 - 1) * dim3;
        int k = 6;

        // Find the top-k probability indices
        std::vector<std::pair<float, int>> prob_and_index;
        const float* start_ptr = tensor_prob + start_index;
        for (int i = 0; i < dim3; ++i) {
            prob_and_index.emplace_back(start_ptr[i], i);
        }

        // Sort in descending order of probability
        std::partial_sort(prob_and_index.begin(), prob_and_index.begin() + k,
                          prob_and_index.end(), std::greater<std::pair<float, int>>());

        // Create a probability distribution over the top-k elements
        std::vector<float> top_k_probs;
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            top_k_probs.push_back(prob_and_index[i].first);
            sum += prob_and_index[i].first;
        }
        std::transform(top_k_probs.begin(), top_k_probs.end(), top_k_probs.begin(),
                       [sum](float prob) { return prob / sum; });

        // Sample an index according to the top-k probability distribution
        std::discrete_distribution<int> distribution(top_k_probs.begin(), top_k_probs.end());
        std::random_device rd;
        std::mt19937 generator(rd());
        int sampled_index = distribution(generator);
        return prob_and_index[sampled_index].second; // Return the sampled token index
    }

    std::vector<Ort::Value> run_inference(SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids) {
        // Building environment for creating ONNX session environment
        // Retrieve the session from the SessionCache object
        Ort::Session& session = sessionCache->inference_session;

        // Create model input layer
        Ort::AllocatorWithDefaultOptions allocator;

        // Print number of model input nodes
        size_t inputCount = session.GetInputCount();

        // Retrieve the input_names
        std::vector<std::string> input_node_names;
        std::vector<std::vector<int64_t>> input_shapes;
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_node_names.push_back(name.get());
            input_shapes.push_back(shape);
        }


        // Retrieve the output_names
        std::vector<std::string> output_node_names;
        std::vector<std::vector<int64_t>> output_shapes;
        size_t outputCount = session.GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            output_node_names.push_back(name.get());
            output_shapes.push_back(shape);
        }

        // Post-process input and output string vector to const char*
        std::vector<const char*> input_node_names_char;
        for (const std::string& name : input_node_names)
            input_node_names_char.push_back(name.c_str());

        std::vector<const char*> output_node_names_char;
        for (const std::string& name : output_node_names)
            output_node_names_char.push_back(name.c_str());


        if (ort_tensors.empty() && !input_ids.empty()){
            int64_t id_col_size = input_ids.size();
            std::vector<int64_t> input_node_dim = {1,id_col_size};
            size_t input_tensor_size = 1 * id_col_size;
            std::vector<int64_t> input_tensor_values(input_tensor_size);
            std::vector<int> test_tensor = input_ids;
            for (size_t i = 0; i < input_tensor_size; i++)
                input_tensor_values[i] = static_cast<int64_t>(test_tensor[i]);

            // Configure memory info
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                                        input_tensor_size, input_node_dim.data(),
                                                                        input_shapes[0].size());
            // Create Ort tensor for inference
            std::vector<Ort::Value> ort_inputs;
            ort_inputs.push_back(std::move(input_tensor));

            auto output_tensors = session.Run(Ort::RunOptions(nullptr),
                                              input_node_names_char.data(), ort_inputs.data(), inputCount,
                                              output_node_names_char.data(), outputCount);
            return output_tensors;
        }
        else if (!ort_tensors.empty() && input_ids.empty()) {
            auto output_tensors = session.Run(Ort::RunOptions(nullptr),
                                              input_node_names_char.data(), ort_tensors.data(), inputCount,
                                              output_node_names_char.data(), outputCount);
            return output_tensors;
        }
    }

    size_t run_inference_with_binary_classification(SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids, int classification) {
        // Building environment for creating ONNX session environment
        // Retrieve the session from the SessionCache object
        Ort::Session& session = sessionCache->inference_session;

        // Create model input layer
        Ort::AllocatorWithDefaultOptions allocator;

        // Print number of model input nodes
        size_t inputCount = session.GetInputCount();

        // Retrieve the input_names
        std::vector<std::string> input_node_names;
        std::vector<std::vector<int64_t>> input_shapes;
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_node_names.push_back(name.get());
            input_shapes.push_back(shape);
        }

        // Retrieve the output_names
        std::vector<std::string> output_node_names;
        std::vector<std::vector<int64_t>> output_shapes;
        size_t outputCount = session.GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            output_node_names.push_back(name.get());
            output_shapes.push_back(shape);
        }

        // Post-process input and output string vector to const char*
        std::vector<const char*> input_node_names_char;
        for (const std::string& name : input_node_names)
            input_node_names_char.push_back(name.c_str());

        std::vector<const char*> output_node_names_char;
        for (const std::string& name : output_node_names)
            output_node_names_char.push_back(name.c_str());

        if (!ort_tensors.empty() && input_ids.empty() && classification==1) {
            auto output_tensors = session.Run(Ort::RunOptions(nullptr),
                                              input_node_names_char.data(), ort_tensors.data(), inputCount,
                                              output_node_names_char.data(), outputCount);
            float *logit = output_tensors.front().GetTensorMutableData<float>();
            size_t classification_result = binary_classify(logit);
            return classification_result;
        }
    }

    int run_inference_with_decoding(SessionCache* sessionCache,
                                    std::vector<Ort::Value> &ort_tensors,
                                    std::vector<int> &input_ids,
                                    int k,
                                    float initial_temp,
                                    int decoding) {
        // Building environment for creating ONNX session environment
        // Retrieve the session from the SessionCache object
        Ort::Session& session = sessionCache->inference_session;

        // Create model input layer
        Ort::AllocatorWithDefaultOptions allocator;

        // Print number of model input nodes
        size_t inputCount = session.GetInputCount();

        // Retrieve the input_names
        std::vector<std::string> input_node_names;
        std::vector<std::vector<int64_t>> input_shapes;
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_node_names.push_back(name.get());
            input_shapes.push_back(shape);
        }

        // Retrieve the output_names
        std::vector<std::string> output_node_names;
        std::vector<std::vector<int64_t>> output_shapes;
        size_t outputCount = session.GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            output_node_names.push_back(name.get());
            output_shapes.push_back(shape);
        }

        // Post-process input and output string vector to const char*
        std::vector<const char*> input_node_names_char;
        for (const std::string& name : input_node_names)
            input_node_names_char.push_back(name.c_str());

        std::vector<const char*> output_node_names_char;
        for (const std::string& name : output_node_names)
            output_node_names_char.push_back(name.c_str());

        if (!ort_tensors.empty() && input_ids.empty() && decoding==1) {
            auto output_tensors = session.Run(Ort::RunOptions(nullptr),
                                              input_node_names_char.data(), ort_tensors.data(), inputCount,
                                              output_node_names_char.data(), outputCount);
//            int decode_id = GreedyDecoding(output_tensors);
            int decode_id = decoding::StaticDecoding(output_tensors, k, initial_temp);
            return decode_id;
        }
    }

    double flop_per_second_estimation(int & model_num_flops, SessionCache* sessionCache, std::vector<Ort::Value> &ort_tensors, std::vector<int> &input_ids){
        // function for running model flop per second estimation based on run_inference_function
        // need the number of model flops from the server

        // Warm up
        const int warm_up_runs = 2;
        for (int i = 0; i < warm_up_runs; ++i) {
            run_inference(sessionCache, ort_tensors, input_ids);
        }
        // Start the timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run inference
        run_inference(sessionCache, ort_tensors, input_ids);

        // End the timer
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate the elapsed time in seconds
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

        // Calculate flops per second
        double flops_per_second = model_num_flops / elapsed_seconds;

        return flops_per_second;
    }


    std::vector<Ort::Value> extractOrtElementsSingle(std::vector<Ort::Value>& data, std::vector<int>& indices) {
        std::vector<Ort::Value> result;

        for (int index : indices) {
            if (index >= 0 && static_cast<size_t>(index) < data.size()) {
                result.push_back(utils::CopyOrtValue(data[index]));  // Pushes a copy of data[index] to the result
            }
        }
        return result;
    }

    std::vector<std::vector<Ort::Value>> extractOrtElementsGroup(std::vector<Ort::Value>& data, std::vector<std::vector<int>>& indicesGroups) {
        std::vector<std::vector<Ort::Value>> result;

        for (const auto& indices : indicesGroups) {
            std::vector<Ort::Value> currentResult;

            for (int index : indices) {
                if (index >= 0 && static_cast<size_t>(index) < data.size()) {
                    currentResult.push_back(utils::CopyOrtValue(data[index]));  // Pushes a copy of data[index] to the currentResult
                }
            }

            result.push_back(utils::CopyOrtValuesVector(currentResult));
        }

        return result;
    }


    std::vector<Ort::Value> combineVectors(const std::vector<Ort::Value>& vec1, const std::vector<Ort::Value>& vec2) {
        std::vector<Ort::Value> combined;
        combined.reserve(vec1.size() + vec2.size());  // Reserve space for efficiency

        // Copy values from vec1
        for (const Ort::Value& val : vec1) {
            combined.push_back(utils::CopyOrtValue(val));
        }

        // Copy values from vec2
        for (const Ort::Value& val : vec2) {
            combined.push_back(utils::CopyOrtValue(val));
        }

        return combined;
    }



}