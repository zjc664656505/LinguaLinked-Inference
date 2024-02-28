//
// Created by Junchen Zhao on 2/26/24.
//

#ifndef DISTRIBUTED_INFERENCE_DEMO_DECODING_H
#define DISTRIBUTED_INFERENCE_DEMO_DECODING_H

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
                       float initial_temp);
}



#endif //DISTRIBUTED_INFERENCE_DEMO_DECODING_H
