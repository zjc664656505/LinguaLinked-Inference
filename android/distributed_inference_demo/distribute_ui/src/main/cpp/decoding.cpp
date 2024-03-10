//
// Created by Junchen Zhao on 2/26/24.
//
#include "decoding.h"
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

namespace decoding {
    int StaticDecoding(std::vector<Ort::Value>& probs,
                       int k,
                       float initial_temp) {
        // TOP-k Sampling
        auto probs_shape = probs.front().GetTensorTypeAndShapeInfo().GetShape();
        int dim2 = probs_shape[1];  // sequence length
        int dim3 = probs_shape[2];  // vocabulary size
        float* tensor_prob = probs.front().GetTensorMutableData<float>();
        int start_index = (dim2 - 1) * dim3;
        float temperature = initial_temp;


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
//            float scaled_prob = std::pow(prob_and_index[i].first, 1/temperature); // Temperature scaling
            float scaled_prob = prob_and_index[i].first;
            top_k_probs.push_back(scaled_prob);
            sum += scaled_prob;
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
}
