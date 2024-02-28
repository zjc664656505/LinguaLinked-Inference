//
// Created by Junchen Zhao on 7/11/23.
//

#include <jni.h>
#include <string>
#include "session_cache.h"
#include "utils.h"
#include "inference.h"
#include "tokenizers_cpp.h"
#include "iostream"
#include "android/log.h"
#include <string>

using tokenizers::Tokenizer;
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_distribute_1ui_SelectionActivity_createSession(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path){
    std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
            utils::JString2String(env, inference_model_path));
    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_distribute_1ui_service_MonitorService_createSession(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path){
    std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
            utils::JString2String(env, inference_model_path));
    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_distribute_1ui_service_MonitorService_releaseSession(
        JNIEnv *env, jobject /* this */,
        jlong session)
{
    if (session == 0) {
        // The pointer is null, and there's nothing to release.
        return;
    }

    auto *session_cache = reinterpret_cast<SessionCache *>(session);

    // Delete the SessionCache object.
    delete session_cache;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_distribute_1ui_SelectionActivity_releaseSession(
        JNIEnv *env, jobject /* this */,
        jlong session)
{
    if (session == 0) {
        // The pointer is null, and there's nothing to release.
        return;
    }

    auto *session_cache = reinterpret_cast<SessionCache *>(session);

    // Delete the SessionCache object.
    delete session_cache;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_distribute_1ui_SelectionActivity_performInferenceMaster(
        JNIEnv *env, jobject /* this */,
        jlong session, jintArray input_ids_j
){
    // function for running inference for master machine
    // should accept input_string in java string type and tokenizer in java long type
    // return a serialized Ort Tensor in ByteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));
    return serialized_tensor_vector;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_distribute_1ui_SelectionActivity_performInferenceWorker(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray data
){
    // running inference on workers
    // should only accepts ort_session and byteArray
    // return a serialized Ort Tenosr in byteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    std::vector<int> input_ids;

    jint length = env->GetArrayLength(data);
    if (length > 0) {
        jbyte* elements = env->GetByteArrayElements(data, nullptr);
        std::vector<char> bytes(elements, elements + length);
        env->ReleaseByteArrayElements(data, elements, 0);
        ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_distribute_1ui_SelectionActivity_binaryClassify(JNIEnv *env, jobject /* this */, jbyteArray data){

    // tensor conversion from java to c++
    jint length = env->GetArrayLength(data);
    std::vector<char> bytes(length);
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    std::vector<Ort::Value> ort_tensor = utils::DeserializeTensorVectorFromBytes(bytes);

    // Error handling for deserialization
    if (ort_tensor.empty()) {
        // Handle the error - replace with your error handling code
        return -1;  // return a value that indicates error
    }

    // retrieve the logit
    float *logit = ort_tensor.front().GetTensorMutableData<float>();
    if (logit == nullptr) {
        // Handle the error - replace with your error handling code
        return -2;  // return a value that indicates error
    }

    size_t prediction = inference::binary_classify(logit);
    if (prediction > std::numeric_limits<jint>::max()) {
        // Handle the error - replace with your error handling code
        return -3;  // return a value that indicates error
    }

    return static_cast<jint>(prediction);
}


// function for creating huggingface tokenizer and convert it to java long type
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_distribute_1ui_SelectionActivity_createHuggingFaceTokenizer(JNIEnv *env, jobject /* this */, jstring tokenizer_file_path_j) {
    const char *tokenizer_file_path_c = env->GetStringUTFChars(tokenizer_file_path_j, nullptr);
    std::string tokenizer_file_path(tokenizer_file_path_c);
    auto tok = inference::HuggingFaceTokenizer(tokenizer_file_path);
    env->ReleaseStringUTFChars(tokenizer_file_path_j, tokenizer_file_path_c);
    return reinterpret_cast<jlong>(tok.release());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_distribute_1ui_SelectionActivity_createSentencePieceTokenizer(JNIEnv *env, jobject /* this */, jstring tokenizer_file_path_j) {
    const char *tokenizer_file_path_c = env->GetStringUTFChars(tokenizer_file_path_j, nullptr);
    std::string tokenizer_file_path(tokenizer_file_path_c);
    auto tok = inference::SentencePieceTokenizer(tokenizer_file_path);
    env->ReleaseStringUTFChars(tokenizer_file_path_j, tokenizer_file_path_c);
    return reinterpret_cast<jlong>(tok.release());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_distribute_1ui_SelectionActivity_tensorSizeDebug(JNIEnv *env, jobject /* this */, jbyteArray data){
    std::vector<Ort::Value> ort_tensors;
    jint length = env->GetArrayLength(data);
    jbyte* elements = env->GetByteArrayElements(data, nullptr);
    std::vector<char> bytes(elements, elements + length);
    env->ReleaseByteArrayElements(data, elements, 0);
    ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    if (ort_tensors.empty()){
        return 1;
    }
    else {
        return 0;
    }
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_distribute_1ui_SelectionActivity_encodeString(JNIEnv *env, jobject /* this */, jstring input_string, jlong tokenizer) {
    std::vector<int> input_ids;

    if(env->GetStringLength(input_string) != 0) {
        std::string input_c_string = utils::JString2String(env, input_string);

        // reinterpret tokenizer to C++ Tokenizer
        auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
        input_ids = inference::Encoding(tokenizer_ptr, input_c_string);
    }

    // convert input_ids to jintArray and return it
    jintArray result = env->NewIntArray(input_ids.size());
    if (result == nullptr) {
        return nullptr; /* out of memory error thrown */
    }
    // move from the temp structure to the java structure
    env->SetIntArrayRegion(result, 0, input_ids.size(), input_ids.data());
    return result;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_example_distribute_1ui_SelectionActivity_decodeID(JNIEnv *env, jobject /* this */, jintArray input_ids_j, jlong tokenizer) {
    std::vector<Ort::Value> ort_tensors;

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    // call decoding
    auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
    std::string output_string = inference::Decoding(tokenizer_ptr, input_ids);

    // convert std::string to jstring and return
    return env->NewStringUTF(output_string.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_distribute_1ui_SelectionActivity_greedyDecoding(JNIEnv *env, jobject /* this */, jbyteArray data) {
    // deserialize remote ort_tensor
    jint length = env->GetArrayLength(data);
    std::vector<char> bytes(length);
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    std::vector<Ort::Value> ort_tensor = utils::DeserializeTensorVectorFromBytes(bytes);

    // call greedy decoding where GreedyDecoding is modified to return a single int value
    int decoded_id = inference::GreedyDecoding(ort_tensor);

    return static_cast<jint>(decoded_id);
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_distribute_1ui_service_MonitorService_modelFlopsPerSecond(JNIEnv *env, jobject /* this */,
                                                                           jint modelFlops, jlong session, jbyteArray data) {
    int model_num_flops = static_cast<int>(modelFlops);
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    std::vector<int> input_ids;

    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp_Native", "flop function start");
    jint length = env->GetArrayLength(data);

    if (length > 0) {
        jbyte* elements = env->GetByteArrayElements(data, nullptr);
        std::vector<char> bytes(elements, elements + length);
        env->ReleaseByteArrayElements(data, elements, 0);
        ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
        __android_log_print(ANDROID_LOG_VERBOSE, "MyApp_Native", "flop completed");
    }

    // get flops per second in double
    double flops_per_second = inference::flop_per_second_estimation(model_num_flops, session_cache, ort_tensors, input_ids);

    return static_cast<jdouble>(flops_per_second);
}


extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_distribute_1ui_SelectionActivity_modelFlopsPerSecond(JNIEnv *env, jobject /* this */,
                                                                      jint modelFlops, jlong session, jbyteArray data) {
    int model_num_flops = static_cast<int>(modelFlops);
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    std::vector<int> input_ids;

    jint length = env->GetArrayLength(data);
    if (length > 0) {
        jbyte* elements = env->GetByteArrayElements(data, nullptr);
        std::vector<char> bytes(elements, elements + length);
        env->ReleaseByteArrayElements(data, elements, 0);
        ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    // get flops per second in double
    double flops_per_second = inference::flop_per_second_estimation(model_num_flops, session_cache, ort_tensors, input_ids);

    return static_cast<jdouble>(flops_per_second);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_distribute_1ui_SelectionActivity_runInferenceMasterResidual(JNIEnv *env, jobject /* this */,
                                                                             jlong session, jintArray input_ids_j,
                                                                             jintArray to_send_seq_indices,
                                                                             jobjectArray to_send_res_indices){

    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    // Convert to_send_seq_indices from jintArray to std::vector<int>
    jsize seq_length = env->GetArrayLength(to_send_seq_indices);
    jint *seq_body = env->GetIntArrayElements(to_send_seq_indices, 0);
    std::vector<int> seq_indices(seq_body, seq_body + seq_length);
    env->ReleaseIntArrayElements(to_send_seq_indices, seq_body, 0);

    // Convert to_send_res_indices from jobjectArray to std::vector<std::vector<int>>
    jsize res_outer_length = env->GetArrayLength(to_send_res_indices);
    std::vector<std::vector<int>> res_indices(res_outer_length);

    if (res_outer_length != 0) {
        for (jsize i = 0; i < res_outer_length; i++) {
            jintArray innerArray = (jintArray) env->GetObjectArrayElement(to_send_res_indices, i);
            jsize inner_length = env->GetArrayLength(innerArray);
            jint *inner_body = env->GetIntArrayElements(innerArray, 0);

            res_indices[i].assign(inner_body, inner_body + inner_length);

            env->ReleaseIntArrayElements(innerArray, inner_body, 0);
            env->DeleteLocalRef(innerArray);
        }
    }

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);
    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    auto seq_result = inference::extractOrtElementsSingle(result, seq_indices);
    jobjectArray resultArray;
    jbyteArray seqByteArr;
    std::vector<char> seq_bytes;

    if (!res_indices.empty()){
        auto residual_result = inference::extractOrtElementsGroup(result, res_indices);
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);

        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create an array of ByteArray objects for the residual results
        jobjectArray resArray = env->NewObjectArray(residual_result.size(), env->FindClass("[B"), NULL);

        for (size_t i = 0; i < residual_result.size(); i++) {
            auto res_bytes = utils::SerializeTensorVectorToBytes(residual_result[i]);
            jbyteArray resByteArr = env->NewByteArray(res_bytes.size());
            env->SetByteArrayRegion(resByteArr, 0, res_bytes.size(), reinterpret_cast<jbyte*>(res_bytes.data()));

            // Now set this byteArray into our resArray
            env->SetObjectArrayElement(resArray, i, resByteArr);
            env->DeleteLocalRef(resByteArr);  // Clean up local reference
        }

        // Create a jobjectArray to hold seqByteArr and resArray
        resultArray = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        env->SetObjectArrayElement(resultArray, 1, resArray);
        env->DeleteLocalRef(resArray);  // Clean up local reference
        return resultArray;
    }
    else {
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);
        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create a jobjectArray to hold just the seq_byte_array
        resultArray = env->NewObjectArray(1, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        // Clean up local references
        env->DeleteLocalRef(seqByteArr);
        return resultArray;
    }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_distribute_1ui_SelectionActivity_runInferenceWorkerResidual(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray sequential_input,
        jobject residual_input,
        jintArray to_send_seq_indices,
        jobjectArray to_send_res_indices
){
    // running inference on workers
    // should only accepts ort_session and byteArray
    // return a serialized Ort Tenosr in byteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    // Convert to_send_seq_indices from jintArray to std::vector<int>
    jsize seq_length = env->GetArrayLength(to_send_seq_indices);
    jint *seq_body = env->GetIntArrayElements(to_send_seq_indices, 0);
    std::vector<int> seq_indices(seq_body, seq_body + seq_length);
    env->ReleaseIntArrayElements(to_send_seq_indices, seq_body, 0);

    // Convert to_send_res_indices from jobjectArray to std::vector<std::vector<int>>
    jsize res_outer_length = env->GetArrayLength(to_send_res_indices);
    std::vector<std::vector<int>> res_indices(res_outer_length);
    if (res_outer_length != 0) {
        for (jsize i = 0; i < res_outer_length; i++) {
            jintArray innerArray = (jintArray) env->GetObjectArrayElement(to_send_res_indices, i);
            jsize inner_length = env->GetArrayLength(innerArray);
            jint *inner_body = env->GetIntArrayElements(innerArray, 0);

            res_indices[i].assign(inner_body, inner_body + inner_length);

            env->ReleaseIntArrayElements(innerArray, inner_body, 0);
            env->DeleteLocalRef(innerArray);
        }
    }

    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;

    jint sequential_length = env->GetArrayLength(sequential_input);
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    // get the arrayList's size
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
    jbyteArray byteArray;
    if (arrayListSize!=0){
        if (arrayListSize == 1) {
            byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, 0);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            residual_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
            env->DeleteLocalRef(byteArray);
        }
        else if (arrayListSize > 1){
            jint i = 0;
            jbyteArray cur_byteArray;
            jbyteArray nxt_byteArray;
            while (i < arrayListSize-1) {
                // Get byte array from ArrayList at index i
                cur_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
                nxt_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i+1);

                // Deserialize byte array to Ort::Value and add to residual_tensors vector
                jint byteArrayLength = env->GetArrayLength(cur_byteArray);
                jbyte* elements = env->GetByteArrayElements(cur_byteArray, nullptr);
                std::vector<char> bytes(elements, elements + byteArrayLength);
                env->ReleaseByteArrayElements(cur_byteArray, elements, 0);

                jint nxt_byteArrayLength = env->GetArrayLength(nxt_byteArray);
                jbyte* nxt_elements = env->GetByteArrayElements(nxt_byteArray, nullptr);
                std::vector<char> nxt_bytes(nxt_elements, nxt_elements + nxt_byteArrayLength);
                env->ReleaseByteArrayElements(nxt_byteArray, nxt_elements, 0);
                std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
                std::vector<Ort::Value> nxt_tensor = utils::DeserializeTensorVectorFromBytes(nxt_bytes);
                if (i==0){
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(tensor, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }
                else{
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(residual_tensors, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }

                env->DeleteLocalRef(cur_byteArray);
                env->DeleteLocalRef(nxt_byteArray);
                i+=1;
            }
        }
        std::vector<Ort::Value> temp_tensor = inference::combineVectors(sequential_tensors, residual_tensors);
        ort_tensors = utils::CopyOrtValuesVector(temp_tensor);
    }
    else {
        ort_tensors = utils::CopyOrtValuesVector(sequential_tensors);

    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    auto seq_result = inference::extractOrtElementsSingle(result, seq_indices);

    jobjectArray resultArray;
    jbyteArray seqByteArr;
    std::vector<char> seq_bytes;

    if (!res_indices.empty()){
        auto residual_result = inference::extractOrtElementsGroup(result, res_indices);
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);

        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create an array of ByteArray objects for the residual results
        jobjectArray resArray = env->NewObjectArray(residual_result.size(), env->FindClass("[B"), NULL);

        for (size_t i = 0; i < residual_result.size(); i++) {
            auto res_bytes = utils::SerializeTensorVectorToBytes(residual_result[i]);
            jbyteArray resByteArr = env->NewByteArray(res_bytes.size());
            env->SetByteArrayRegion(resByteArr, 0, res_bytes.size(), reinterpret_cast<jbyte*>(res_bytes.data()));

            // Now set this byteArray into our resArray
            env->SetObjectArrayElement(resArray, i, resByteArr);
            env->DeleteLocalRef(resByteArr);  // Clean up local reference
        }

        // Create a jobjectArray to hold seqByteArr and resArray
        resultArray = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        env->SetObjectArrayElement(resultArray, 1, resArray);
        env->DeleteLocalRef(resArray);  // Clean up local reference
        return resultArray;
    }
    else {
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);
        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create a jobjectArray to hold just the seq_byte_array
        resultArray = env->NewObjectArray(1, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        // Clean up local references
        env->DeleteLocalRef(seqByteArr);
        return resultArray;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_distribute_1ui_SelectionActivity_runInferenceWorkerResidualLast(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray sequential_input,
        jobject residual_input
){
    auto* session_cache = reinterpret_cast<SessionCache *>(session);
    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;

    jint sequential_length = env->GetArrayLength(sequential_input);
    //
    // meet error after calling __android_log_print
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    // get the arrayList's size
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
    jbyteArray byteArray;

    if (arrayListSize!=0){
        if (arrayListSize == 1) {
            byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, 0);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            residual_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
            env->DeleteLocalRef(byteArray);
        }
        else if (arrayListSize > 1){
            jint i = 0;
            jbyteArray cur_byteArray;
            jbyteArray nxt_byteArray;
            while (i < arrayListSize-1) {
                // Get byte array from ArrayList at index i
                cur_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
                nxt_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i+1);


                // Deserialize byte array to Ort::Value and add to residual_tensors vector
                jint byteArrayLength = env->GetArrayLength(cur_byteArray);
                jbyte* elements = env->GetByteArrayElements(cur_byteArray, nullptr);
                std::vector<char> bytes(elements, elements + byteArrayLength);
                env->ReleaseByteArrayElements(cur_byteArray, elements, 0);

                jint nxt_byteArrayLength = env->GetArrayLength(nxt_byteArray);
                jbyte* nxt_elements = env->GetByteArrayElements(nxt_byteArray, nullptr);
                std::vector<char> nxt_bytes(nxt_elements, nxt_elements + nxt_byteArrayLength);
                env->ReleaseByteArrayElements(nxt_byteArray, nxt_elements, 0);


                std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
                std::vector<Ort::Value> nxt_tensor = utils::DeserializeTensorVectorFromBytes(nxt_bytes);
                if (i==0){
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(tensor, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }
                else{
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(residual_tensors, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }

                env->DeleteLocalRef(cur_byteArray);
                env->DeleteLocalRef(nxt_byteArray);
                i += 1;
            }
        }

        std::vector<Ort::Value> temp_tensor = inference::combineVectors(sequential_tensors, residual_tensors);
        ort_tensors = utils::CopyOrtValuesVector(temp_tensor);
    }
    else {
        ort_tensors = utils::CopyOrtValuesVector(sequential_tensors);
    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);
    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}



//====================================================================================================
// SecureConnection JNI methods
//====================================================================================================


extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_SecureConnection_Client_createSession(JNIEnv *env, jobject thiz,
                                                       jstring inference_model_path) {
    std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
            utils::JString2String(env, inference_model_path));
    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_SecureConnection_Client_releaseSession(
        JNIEnv *env, jobject /* this */,
        jlong session)
{
    if (session == 0) {
        // The pointer is null, and there's nothing to release.
        return;
    }

    auto *session_cache = reinterpret_cast<SessionCache *>(session);

    // Delete the SessionCache object.
    delete session_cache;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_SecureConnection_Communication_performInferenceMaster(
        JNIEnv *env, jobject /* this */,
        jlong session, jintArray input_ids_j
){
    // function for running inference for master machine
    // should accept input_string in java string type and tokenizer in java long type
    // return a serialized Ort Tensor in ByteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));
    return serialized_tensor_vector;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_SecureConnection_Communication_performInferenceWorker(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray data
){
    // running inference on workers
    // should only accepts ort_session and byteArray
    // return a serialized Ort Tenosr in byteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    std::vector<int> input_ids;

    jint length = env->GetArrayLength(data);
    if (length > 0) {
        jbyte* elements = env->GetByteArrayElements(data, nullptr);
        std::vector<char> bytes(elements, elements + length);
        env->ReleaseByteArrayElements(data, elements, 0);
        ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_distribute_1ui_BackgroundService_binaryClassify(JNIEnv *env, jobject /* this */, jbyteArray data){

    // tensor conversion from java to c++
    jint length = env->GetArrayLength(data);
    std::vector<char> bytes(length);
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    std::vector<Ort::Value> ort_tensor = utils::DeserializeTensorVectorFromBytes(bytes);

    // Error handling for deserialization
    if (ort_tensor.empty()) {
        // Handle the error - replace with your error handling code
        return -1;  // return a value that indicates error
    }

    // retrieve the logit
    float *logit = ort_tensor.front().GetTensorMutableData<float>();
    if (logit == nullptr) {
        // Handle the error - replace with your error handling code
        return -2;  // return a value that indicates error
    }

    size_t prediction = inference::binary_classify(logit);
    if (prediction > std::numeric_limits<jint>::max()) {
        // Handle the error - replace with your error handling code
        return -3;  // return a value that indicates error
    }

    return static_cast<jint>(prediction);
}


// function for creating huggingface tokenizer and convert it to java long type
extern "C" [[maybe_unused]] JNIEXPORT jlong JNICALL
Java_com_example_SecureConnection_Client_createHuggingFaceTokenizer(JNIEnv *env, jobject /* this */, jstring tokenizer_file_path_j) {
    const char *tokenizer_file_path_c = env->GetStringUTFChars(tokenizer_file_path_j, nullptr);
    std::string tokenizer_file_path(tokenizer_file_path_c);
    auto tok = inference::HuggingFaceTokenizer(tokenizer_file_path);
    env->ReleaseStringUTFChars(tokenizer_file_path_j, tokenizer_file_path_c);
    return reinterpret_cast<jlong>(tok.release());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_SecureConnection_Client_createSentencePieceTokenizer(JNIEnv *env, jobject /* this */, jstring tokenizer_file_path_j) {
    const char *tokenizer_file_path_c = env->GetStringUTFChars(tokenizer_file_path_j, nullptr);
    std::string tokenizer_file_path(tokenizer_file_path_c);
    auto tok = inference::SentencePieceTokenizer(tokenizer_file_path);
    env->ReleaseStringUTFChars(tokenizer_file_path_j, tokenizer_file_path_c);
    return reinterpret_cast<jlong>(tok.release());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_SecureConnection_Communication_tensorSizeDebug(JNIEnv *env, jobject /* this */, jbyteArray data){
    std::vector<Ort::Value> ort_tensors;
    jint length = env->GetArrayLength(data);
    jbyte* elements = env->GetByteArrayElements(data, nullptr);
    std::vector<char> bytes(elements, elements + length);
    env->ReleaseByteArrayElements(data, elements, 0);
    ort_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    if (ort_tensors.empty()){
        return 1;
    }
    else {
        return 0;
    }
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_SecureConnection_Communication_encodeString(JNIEnv *env, jobject /* this */, jstring input_string, jlong tokenizer) {
    std::vector<int> input_ids;

    if(env->GetStringLength(input_string) != 0) {
        std::string input_c_string = utils::JString2String(env, input_string);

        // reinterpret tokenizer to C++ Tokenizer
        auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
        input_ids = inference::Encoding(tokenizer_ptr, input_c_string);
    }

    // convert input_ids to jintArray and return it
    jintArray result = env->NewIntArray(input_ids.size());
    if (result == nullptr) {
        return nullptr; /* out of memory error thrown */
    }
    // move from the temp structure to the java structure
    env->SetIntArrayRegion(result, 0, input_ids.size(), input_ids.data());
    return result;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_example_SecureConnection_Communication_decodeID(JNIEnv *env, jobject /* this */, jintArray input_ids_j, jlong tokenizer) {
    std::vector<Ort::Value> ort_tensors;

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    // call decoding
    auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
    std::string output_string = inference::Decoding(tokenizer_ptr, input_ids);

    // convert std::string to jstring and return
    return env->NewStringUTF(output_string.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_SecureConnection_Communication_greedyDecoding(JNIEnv *env, jobject /* this */, jbyteArray data) {
    // deserialize remote ort_tensor
    jint length = env->GetArrayLength(data);
    std::vector<char> bytes(length);
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    std::vector<Ort::Value> ort_tensor = utils::DeserializeTensorVectorFromBytes(bytes);

    // call greedy decoding where GreedyDecoding is modified to return a single int value
    int decoded_id = inference::GreedyDecoding(ort_tensor);

    return static_cast<jint>(decoded_id);
}


extern "C" JNIEXPORT jint JNICALL
Java_com_example_SecureConnection_Communication_binaryClassify(JNIEnv *env, jobject /* this */, jbyteArray data){

    // tensor conversion from java to c++
    jint length = env->GetArrayLength(data);
    std::vector<char> bytes(length);
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    std::vector<Ort::Value> ort_tensor = utils::DeserializeTensorVectorFromBytes(bytes);

    // Error handling for deserialization
    if (ort_tensor.empty()) {
        // Handle the error - replace with your error handling code
        return -1;  // return a value that indicates error
    }

    // retrieve the logit
    float *logit = ort_tensor.front().GetTensorMutableData<float>();
    if (logit == nullptr) {
        // Handle the error - replace with your error handling code
        return -2;  // return a value that indicates error
    }

    size_t prediction = inference::binary_classify(logit);
    if (prediction > std::numeric_limits<jint>::max()) {
        // Handle the error - replace with your error handling code
        return -3;  // return a value that indicates error
    }

    return static_cast<jint>(prediction);
}


extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_SecureConnection_Communication_modelFlopsPerSecond(JNIEnv *env, jobject /* this */,
                                                                    jint modelFlops, jlong session, jintArray input_ids_j) {
    int model_num_flops = static_cast<int>(modelFlops);
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);


    // get flops per second in double
    double flops_per_second = inference::flop_per_second_estimation(model_num_flops, session_cache, ort_tensors, input_ids);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);
    return static_cast<jdouble>(flops_per_second);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_SecureConnection_Communication_runInferenceMasterResidual(JNIEnv *env, jobject /* this */,
                                                                           jlong session, jintArray input_ids_j,
                                                                           jintArray to_send_seq_indices,
                                                                           jobjectArray to_send_res_indices){

    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    std::vector<Ort::Value> ort_tensors;
    // Convert to_send_seq_indices from jintArray to std::vector<int>
    jsize seq_length = env->GetArrayLength(to_send_seq_indices);
    jint *seq_body = env->GetIntArrayElements(to_send_seq_indices, 0);
    std::vector<int> seq_indices(seq_body, seq_body + seq_length);
    env->ReleaseIntArrayElements(to_send_seq_indices, seq_body, 0);

    // Convert to_send_res_indices from jobjectArray to std::vector<std::vector<int>>
    jsize res_outer_length = env->GetArrayLength(to_send_res_indices);
    std::vector<std::vector<int>> res_indices(res_outer_length);

    if (res_outer_length != 0) {
        for (jsize i = 0; i < res_outer_length; i++) {
            jintArray innerArray = (jintArray) env->GetObjectArrayElement(to_send_res_indices, i);
            jsize inner_length = env->GetArrayLength(innerArray);
            jint *inner_body = env->GetIntArrayElements(innerArray, 0);

            res_indices[i].assign(inner_body, inner_body + inner_length);

            env->ReleaseIntArrayElements(innerArray, inner_body, 0);
            env->DeleteLocalRef(innerArray);
        }
    }

    // Converting jintArray to std::vector<int>
    jsize length = env->GetArrayLength(input_ids_j);
    jint *body = env->GetIntArrayElements(input_ids_j, 0);
    std::vector<int> input_ids(body, body + length);
    env->ReleaseIntArrayElements(input_ids_j, body, 0);

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);
    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    auto seq_result = inference::extractOrtElementsSingle(result, seq_indices);
    jobjectArray resultArray;
    jbyteArray seqByteArr;
    std::vector<char> seq_bytes;

    if (!res_indices.empty()){
        auto residual_result = inference::extractOrtElementsGroup(result, res_indices);
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);

        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create an array of ByteArray objects for the residual results
        jobjectArray resArray = env->NewObjectArray(residual_result.size(), env->FindClass("[B"), NULL);

        for (size_t i = 0; i < residual_result.size(); i++) {
            auto res_bytes = utils::SerializeTensorVectorToBytes(residual_result[i]);
            jbyteArray resByteArr = env->NewByteArray(res_bytes.size());
            env->SetByteArrayRegion(resByteArr, 0, res_bytes.size(), reinterpret_cast<jbyte*>(res_bytes.data()));

            // Now set this byteArray into our resArray
            env->SetObjectArrayElement(resArray, i, resByteArr);
            env->DeleteLocalRef(resByteArr);  // Clean up local reference
        }

        // Create a jobjectArray to hold seqByteArr and resArray
        resultArray = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        env->SetObjectArrayElement(resultArray, 1, resArray);
        env->DeleteLocalRef(resArray);  // Clean up local reference
        return resultArray;
    }
    else {
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);
        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create a jobjectArray to hold just the seq_byte_array
        resultArray = env->NewObjectArray(1, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        // Clean up local references
        env->DeleteLocalRef(seqByteArr);
        return resultArray;
    }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_SecureConnection_Communication_runInferenceWorkerResidual(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray sequential_input,
        jobject residual_input,
        jintArray to_send_seq_indices,
        jobjectArray to_send_res_indices
){
    // running inference on workers
    // should only accepts ort_session and byteArray
    // return a serialized Ort Tenosr in byteArray
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    // Convert to_send_seq_indices from jintArray to std::vector<int>
    jsize seq_length = env->GetArrayLength(to_send_seq_indices);
    jint *seq_body = env->GetIntArrayElements(to_send_seq_indices, 0);
    std::vector<int> seq_indices(seq_body, seq_body + seq_length);
    env->ReleaseIntArrayElements(to_send_seq_indices, seq_body, 0);

    // Convert to_send_res_indices from jobjectArray to std::vector<std::vector<int>>
    jsize res_outer_length = env->GetArrayLength(to_send_res_indices);
    std::vector<std::vector<int>> res_indices(res_outer_length);
    if (res_outer_length != 0) {
        for (jsize i = 0; i < res_outer_length; i++) {
            jintArray innerArray = (jintArray) env->GetObjectArrayElement(to_send_res_indices, i);
            jsize inner_length = env->GetArrayLength(innerArray);
            jint *inner_body = env->GetIntArrayElements(innerArray, 0);

            res_indices[i].assign(inner_body, inner_body + inner_length);

            env->ReleaseIntArrayElements(innerArray, inner_body, 0);
            env->DeleteLocalRef(innerArray);
        }
    }

    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;

    jint sequential_length = env->GetArrayLength(sequential_input);
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    // get the arrayList's size
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
    jbyteArray byteArray;
    if (arrayListSize!=0){
        if (arrayListSize == 1) {
            byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, 0);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            residual_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
            env->DeleteLocalRef(byteArray);
        }
        else if (arrayListSize > 1){
            jint i = 0;
            jbyteArray cur_byteArray;
            jbyteArray nxt_byteArray;
            while (i < arrayListSize-1) {
                // Get byte array from ArrayList at index i
                cur_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
                nxt_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i+1);

                // Deserialize byte array to Ort::Value and add to residual_tensors vector
                jint byteArrayLength = env->GetArrayLength(cur_byteArray);
                jbyte* elements = env->GetByteArrayElements(cur_byteArray, nullptr);
                std::vector<char> bytes(elements, elements + byteArrayLength);
                env->ReleaseByteArrayElements(cur_byteArray, elements, 0);

                jint nxt_byteArrayLength = env->GetArrayLength(nxt_byteArray);
                jbyte* nxt_elements = env->GetByteArrayElements(nxt_byteArray, nullptr);
                std::vector<char> nxt_bytes(nxt_elements, nxt_elements + nxt_byteArrayLength);
                env->ReleaseByteArrayElements(nxt_byteArray, nxt_elements, 0);
                std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
                std::vector<Ort::Value> nxt_tensor = utils::DeserializeTensorVectorFromBytes(nxt_bytes);
                if (i==0){
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(tensor, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }
                else{
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(residual_tensors, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }

                env->DeleteLocalRef(cur_byteArray);
                env->DeleteLocalRef(nxt_byteArray);
                i+=1;
            }
        }
        std::vector<Ort::Value> temp_tensor = inference::combineVectors(sequential_tensors, residual_tensors);
        ort_tensors = utils::CopyOrtValuesVector(temp_tensor);
    }
    else {
        ort_tensors = utils::CopyOrtValuesVector(sequential_tensors);

    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);

    if (result.empty()){
        throw std::runtime_error("inference master has empty ort_tensor logit.");
    }

    auto seq_result = inference::extractOrtElementsSingle(result, seq_indices);

    jobjectArray resultArray;
    jbyteArray seqByteArr;
    std::vector<char> seq_bytes;

    if (!res_indices.empty()){
        auto residual_result = inference::extractOrtElementsGroup(result, res_indices);
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);

        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create an array of ByteArray objects for the residual results
        jobjectArray resArray = env->NewObjectArray(residual_result.size(), env->FindClass("[B"), NULL);

        for (size_t i = 0; i < residual_result.size(); i++) {
            auto res_bytes = utils::SerializeTensorVectorToBytes(residual_result[i]);
            jbyteArray resByteArr = env->NewByteArray(res_bytes.size());
            env->SetByteArrayRegion(resByteArr, 0, res_bytes.size(), reinterpret_cast<jbyte*>(res_bytes.data()));

            // Now set this byteArray into our resArray
            env->SetObjectArrayElement(resArray, i, resByteArr);
            env->DeleteLocalRef(resByteArr);  // Clean up local reference
        }

        // Create a jobjectArray to hold seqByteArr and resArray
        resultArray = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        env->SetObjectArrayElement(resultArray, 1, resArray);
        env->DeleteLocalRef(resArray);  // Clean up local reference
        return resultArray;
    }
    else {
        seq_bytes = utils::SerializeTensorVectorToBytes(seq_result);
        seqByteArr = env->NewByteArray(seq_bytes.size());
        env->SetByteArrayRegion(seqByteArr, 0, seq_bytes.size(), reinterpret_cast<jbyte*>(seq_bytes.data()));

        // Create a jobjectArray to hold just the seq_byte_array
        resultArray = env->NewObjectArray(1, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(resultArray, 0, seqByteArr);
        // Clean up local references
        env->DeleteLocalRef(seqByteArr);
        return resultArray;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_SecureConnection_Communication_runInferenceWorkerResidualLast(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray sequential_input,
        jobject residual_input
){
    auto* session_cache = reinterpret_cast<SessionCache *>(session);
    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;

    jint sequential_length = env->GetArrayLength(sequential_input);
    //
    // meet error after calling __android_log_print
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
    }

    // get the arrayList's size
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
    jbyteArray byteArray;

    if (arrayListSize!=0){
        if (arrayListSize == 1) {
            byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, 0);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            residual_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
            env->DeleteLocalRef(byteArray);
        }
        else if (arrayListSize > 1){
            jint i = 0;
            jbyteArray cur_byteArray;
            jbyteArray nxt_byteArray;
            while (i < arrayListSize-1) {
                // Get byte array from ArrayList at index i
                cur_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
                nxt_byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i+1);


                // Deserialize byte array to Ort::Value and add to residual_tensors vector
                jint byteArrayLength = env->GetArrayLength(cur_byteArray);
                jbyte* elements = env->GetByteArrayElements(cur_byteArray, nullptr);
                std::vector<char> bytes(elements, elements + byteArrayLength);
                env->ReleaseByteArrayElements(cur_byteArray, elements, 0);

                jint nxt_byteArrayLength = env->GetArrayLength(nxt_byteArray);
                jbyte* nxt_elements = env->GetByteArrayElements(nxt_byteArray, nullptr);
                std::vector<char> nxt_bytes(nxt_elements, nxt_elements + nxt_byteArrayLength);
                env->ReleaseByteArrayElements(nxt_byteArray, nxt_elements, 0);


                std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
                std::vector<Ort::Value> nxt_tensor = utils::DeserializeTensorVectorFromBytes(nxt_bytes);
                if (i==0){
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(tensor, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }
                else{
                    std::vector<Ort::Value> temp_tensor = inference::combineVectors(residual_tensors, nxt_tensor);
                    residual_tensors = utils::CopyOrtValuesVector(temp_tensor);
                }

                env->DeleteLocalRef(cur_byteArray);
                env->DeleteLocalRef(nxt_byteArray);
                i += 1;
            }
        }

        std::vector<Ort::Value> temp_tensor = inference::combineVectors(sequential_tensors, residual_tensors);
        ort_tensors = utils::CopyOrtValuesVector(temp_tensor);
    }
    else {
        ort_tensors = utils::CopyOrtValuesVector(sequential_tensors);
    }

    auto result = inference::run_inference(session_cache, ort_tensors, input_ids);
    std::vector<char> bytes = utils::SerializeTensorVectorToBytes(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_SecureConnection_Communication_releaseSession(JNIEnv *env, jobject thiz,
                                                               jlong session) {
    if (session == 0) {
        // The pointer is null, and there's nothing to release.
        return;
    }

    auto *session_cache = reinterpret_cast<SessionCache *>(session);

    // Delete the SessionCache object.
    delete session_cache;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_SecureConnection_Communication_runInferenceWorkerResidualLastClassification(
        JNIEnv *env, jobject /* this */,
        jlong session, jbyteArray sequential_input,
        jobject residual_input
){
    auto* session_cache = reinterpret_cast<SessionCache *>(session);
    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;

    jint sequential_length = env->GetArrayLength(sequential_input);
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
        // Clear bytes vector to free memory
        bytes.clear();
        bytes.shrink_to_fit();
    }

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");

    if (arrayListSize > 0) {
        for (jint i = 0; i < arrayListSize; ++i) {
            jbyteArray byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            env->DeleteLocalRef(byteArray);

            // Combine tensors directly without unnecessary copies
            std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
            residual_tensors = inference::combineVectors(residual_tensors, tensor);

            // Clear bytes vector to free memory
            bytes.clear();
            bytes.shrink_to_fit();
        }
        ort_tensors = inference::combineVectors(sequential_tensors, residual_tensors);
    } else {
        ort_tensors = std::move(sequential_tensors);
    }

    auto result = inference::run_inference_with_binary_classification(session_cache, ort_tensors, input_ids, 1);
    std::vector<char> bytes = utils::SerializeInt(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    if (serialized_tensor_vector == nullptr) {
        // Handle memory allocation failure
        return nullptr;
    }
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_SecureConnection_Communication_runInferenceWorkerResidualLastGeneration(
        JNIEnv *env, jobject /* this */,
        jlong session,
        jbyteArray sequential_input,
        jobject residual_input,
        jint k,
        jfloat init_temp,
        jfloat final_temp,
        jint max_len,
        jint current_gen
){
    auto* session_cache = reinterpret_cast<SessionCache *>(session);
    std::vector<Ort::Value> ort_tensors;
    std::vector<Ort::Value> sequential_tensors;
    std::vector<Ort::Value> residual_tensors;
    std::vector<int> input_ids;
    int top_k = k;
    float initial_temp = init_temp;

    jint sequential_length = env->GetArrayLength(sequential_input);
    if (sequential_length > 0) {
        jbyte* elements = env->GetByteArrayElements(sequential_input, nullptr);
        std::vector<char> bytes(elements, elements + sequential_length);
        env->ReleaseByteArrayElements(sequential_input, elements, 0);
        sequential_tensors = utils::DeserializeTensorVectorFromBytes(bytes);
        // Clear bytes vector to free memory
        bytes.clear();
        bytes.shrink_to_fit();
    }

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListSizeMethodID = env->GetMethodID(arrayListClass, "size", "()I");
    jint arrayListSize = env->CallIntMethod(residual_input, arrayListSizeMethodID);
    jmethodID arrayListGetMethodID = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");

    if (arrayListSize > 0) {
        for (jint i = 0; i < arrayListSize; ++i) {
            jbyteArray byteArray = (jbyteArray) env->CallObjectMethod(residual_input, arrayListGetMethodID, i);
            jint byteArrayLength = env->GetArrayLength(byteArray);
            jbyte* elements = env->GetByteArrayElements(byteArray, nullptr);
            std::vector<char> bytes(elements, elements + byteArrayLength);
            env->ReleaseByteArrayElements(byteArray, elements, 0);
            env->DeleteLocalRef(byteArray);

            // Combine tensors directly without unnecessary copies
            std::vector<Ort::Value> tensor = utils::DeserializeTensorVectorFromBytes(bytes);
            residual_tensors = inference::combineVectors(residual_tensors, tensor);

            // Clear bytes vector to free memory
            bytes.clear();
            bytes.shrink_to_fit();
        }
        ort_tensors = inference::combineVectors(sequential_tensors, residual_tensors);
    } else {
        ort_tensors = std::move(sequential_tensors);
    }

    auto result = inference::run_inference_with_decoding(session_cache,
                                                         ort_tensors,
                                                         input_ids,
                                                         top_k,
                                                         initial_temp,
                                                         1);

    std::vector<char> bytes = utils::SerializeInt(result);

    jbyteArray serialized_tensor_vector = env->NewByteArray(bytes.size());
    if (serialized_tensor_vector == nullptr) {
        // Handle memory allocation failure
        return nullptr;
    }
    env->SetByteArrayRegion(serialized_tensor_vector, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));

    return serialized_tensor_vector;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_SecureConnection_Communication_deserializeInt(JNIEnv *env, jobject thiz,
                                                               jbyteArray decode_id) {
    jint resultInt = 0;
    jint bytearray_length = env->GetArrayLength(decode_id);

    if (bytearray_length != sizeof(int)) {
        // Throw an exception or handle the error as needed
        return 0;  // Or another error value
    }

    if (bytearray_length > 0) {
        jbyte* elements = env->GetByteArrayElements(decode_id, nullptr);
        if (elements != nullptr) {
            std::vector<char> bytes(elements, elements + bytearray_length);
            env->ReleaseByteArrayElements(decode_id, elements, JNI_ABORT);

            resultInt = utils::DeserializeInt(bytes);

            // Clear bytes vector to free memory
            bytes.clear();
            bytes.shrink_to_fit();
        }
    }
    return resultInt;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_SecureConnection_Communication_TokenToID(JNIEnv *env, jobject thiz,
        jstring token, jlong tokenizer){
    const char* rawString = env->GetStringUTFChars(token, nullptr);
    std::string cppString(rawString);
    auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
    env->ReleaseStringUTFChars(token, rawString);
    return tokenizer_ptr->TokenToId(cppString);

}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_SecureConnection_Communication_EosCheck(JNIEnv *env, jobject thiz,
                                                          jbyteArray output, jlong tokenizer){
    jsize length = env->GetArrayLength(output);
    jbyte* bytes = env->GetByteArrayElements(output, NULL);
    int value;
    memcpy(&value, bytes, sizeof(value));
    auto tokenizer_ptr = reinterpret_cast<Tokenizer*>(tokenizer);
    return tokenizer_ptr->TokenToId("</s>") == value;
}




