//
// Created by Junchen Zhao on 7/9/23.
//

#ifndef DISTRIBUTED_INFERENCE_DEMO_UTILS_H
#define DISTRIBUTED_INFERENCE_DEMO_UTILS_H

#include <string>
#include <jni.h>
#include "onnxruntime_cxx_api.h"

namespace utils {

    // Convert jstring to std::string
    // More specifically, jstring is the string represented in java type
    // this function is used to convert java string to string in C++
    std::string JString2String(JNIEnv *env, jstring jStr);
    std::vector<char> SerializeTensorVectorToBytes(const std::vector<Ort::Value>& tensors);
    std::vector<Ort::Value> DeserializeTensorVectorFromBytes(const std::vector<char>& bytes);
    Ort::Value CopyOrtValue(const Ort::Value& original);
    std::vector<Ort::Value> CopyOrtValuesVector(const std::vector<Ort::Value>& originalVector);
    std::vector<char> SerializeInt(int value);
    int DeserializeInt(const std::vector<char>& byteArray);

} // utils

#endif //DISTRIBUTED_INFERENCE_DEMO_UTILS_H
