//
// Created by Junchen Zhao on 7/9/23.
//

#ifndef DISTRIBUTED_INFERENCE_DEMO_SESSION_CACHE_H
#define DISTRIBUTED_INFERENCE_DEMO_SESSION_CACHE_H

#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"

// class for loading inference model
struct ArtifactPaths {
    std::string inference_model_path;

    ArtifactPaths(const std::string &inference_model_path) :
            inference_model_path(inference_model_path) {}
};

// class for creating onnx session
struct SessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    Ort::Session inference_session;

    SessionCache(const std::string &inference_model_path) :
            artifact_paths(inference_model_path),
            ort_env(ORT_LOGGING_LEVEL_WARNING, "distributed inference demo"),
            session_options(),
            inference_session(ort_env, inference_model_path.c_str(), session_options) {
        uint32_t nnapi_flag = 0;
        nnapi_flag |= NNAPI_FLAG_CPU_DISABLED;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flag));
    }
};
#endif //DISTRIBUTED_INFERENCE_DEMO_SESSION_CACHE_H
