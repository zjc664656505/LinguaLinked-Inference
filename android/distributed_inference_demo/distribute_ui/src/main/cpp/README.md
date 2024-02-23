# C++ Build Environment Setup
1. Making sure you have installed latest CMake on your system. Please check the version of CMake `cmake --version`. In this building environment, we require you to have `cmake version 3.26.3` at least.
2. Making sure you have installed latest ninja on your system. Please check the version of Ninja `ninja --version`. In this building environment, we require you to have `ninja version 11.1.1` at least.
3. Double check whether your `NDK` is configured in your android studio. If not, go to `Tools` -> `SDK Manager` -> `SDK Tools` in android studio, and select `NDK (Side by side)` to install the `NDK` to your android development environment.

# Configure Tokenizer

## rust setup
1. Before setting up the tokenizer, making sure you have latest `rust` installed. Go to webpage `https://www.rust-lang.org/tools/install` to install it.
2. Once you have installed `rust`, making sure to install corresponding rust target. For example, if our target architecture is `aarch64-linux-android`, then do `rustup target add aarch64-linux-android`.


## tokenizer setup
1. Making sure current directory is `.../src/main/cpp/`.
2. If `tokenizers-cpp` folder exists, remove it first by `rm -f tokenizers-cpp`. Then, clone `tokenizers-cpp` repo `git clone https://github.com/mlc-ai/tokenizers-cpp.git`.
3. `cd tokenizers-cpp` and remove the empty sentencepiece directory `rm -rf sentencepiece`.
4. Clone `sentencepiece` repo under the `tokenizers-cpp` directory `git clone https://github.com/google/sentencepiece.git`.
5. Once cloned the `sentencepiece`, go to the `sentencepiece/src/CMakeLists.txt`.
6. In the `sentencepiece/src/CMakeLists.txt`, change the line ~ line 230 `target_link_libraries(sentencepiece-static INTERFACE ${SPM_LIBS})` to `target_link_libraries(sentencepiece-static INTERFACE ${SPM_LIBS} log)`.

# NNAPI Accerlation
The Neural Networks API (NNAPI) is part of the Android operating system, introduced in Android 8.1, and it's designed to provide high performance and efficient computation for machine learning inference, utilizing hardware accelerators when available.
Running ONNX inference with Android's NNAPI has several benefits:

1. Performance: The NNAPI is designed to take advantage of underlying hardware acceleration (like GPUs or DSPs), when available. This can lead to significant speedups in executing the model's inference compared to CPU-only computation.

2. Efficiency: Using hardware acceleration often leads to increased power efficiency, which is a crucial factor for mobile or embedded devices, leading to better battery life.

3. Versatility: By exporting your model to ONNX, you aren't tied to any specific framework for inference. This makes your pipeline more versatile and adaptable to changes.

4. Standardization: Both ONNX and NNAPI are widely supported, which makes your models and applications more interoperable with other systems and devices.

## Four options of running NNAPI
1. `NNAPI_FLAG_USE_FP16`: Use fp16 relaxation in NNAPI EP. This may improve performance but can also reduce accuracy due to the lower precision. For choosing this option, go to `/cpp/session_cache.h` and change the `nnapi_flags` to `nnapi_flags |= NNAPI_FLAG_USE_FP16;`.
2. `NNAPI_FLAG_USE_NCHW`: Use the NCHW layout in NNAPI EP. This is only available for Android API level 29 and higher. Please note that for now, NNAPI might have worse performance using NCHW compared to using NHWC. `nnapi_flags |= NNAPI_FLAG_USE_NCHW;`
3. `NNAPI_FLAG_CPU_DISABLED` (*Current Option): Prevent NNAPI from using CPU devices. NNAPI is more efficient using GPU or NPU for execution, however NNAPI might fall back to its CPU implementation for operations that are not supported by GPU/NPU. The CPU implementation of NNAPI (which is called nnapi-reference) is often less efficient than the optimized versions of the operation of ORT. Due to this, it may be advantageous to disable the NNAPI CPU fallback and handle execution using ORT kernels.  For some models, if NNAPI would use CPU to execute an operation, and this flag is set, the execution of the model may fall back to ORT kernels. 

   4. This option is only available for Android API level 29 and higher, and will be ignored for Android API level 28 and lower. 

   5. For NNAPI device assignments, see https://developer.android.com/ndk/guides/neuralnetworks#device-assignment. 

   6. For NNAPI CPU fallback, see https://developer.android.com/ndk/guides/neuralnetworks#cpu-fallback.

   7. `nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;`
      
8. `NNAPI_FLAG_CPU_ONLY`: Using CPU only in NNAPI EP, this may decrease the perf but will provide reference output value without precision loss, which is useful for validation. This option is only available for Android API level 29 and higher, and will be ignored for Android API level 28 and lower. `nnapi_flags |= NNAPI_FLAG_CPU_ONLY;`

# Build Shared Library (.so) file for Java reference
If you are intend to use this C++ library for Java reference.

1. Go to file `.../src/main/cpp/native-lib.cpp`, change the JNI function class names to the one you need. Currently, in this example, we defined JNI interface for `MainActivity.kt` to refer.

   2. For example, the current JNI interface for `createSession` is defined shown below
   ```cpp
   extern "C" JNIEXPORT jlong JNICALL
   Java_com_example_test1_MainActivity_createSession(
           JNIEnv *env, jobject /* this */,
           jstring inference_model_path){
       std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
               utils::JString2String(env, inference_model_path));
       return reinterpret_cast<long>(session_cache.release());
   }
   ```
   
   3. Assume that you want to refer this JNI interface function in a java class `com.example.test1/TcpConnection.java`, then you can define a new JNI interface `createSession` function shown below
   ```cpp
   extern "C" JNIEXPORT jlong JNICALL
   Java_com_example_test1_TcpConnection_createSession(
           JNIEnv *env, jobject /* this */,
           jstring inference_model_path){
       std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
               utils::JString2String(env, inference_model_path));
       return reinterpret_cast<long>(session_cache.release());
   }
   ```
   
3. Clean your project build and rebuild the project in android studio.
4. Once you successfully build your project, go to the folder `.../app/build/intermediate/cxx/debug/obj/<ABI> (like armeabi-v7a, arm64-v8a, x86, x86_64, etc.)/` and you should be able to find a shared library file `libdistributed_inference_demo.so`.
5. You can load the shared library file in your java class by
```java
// TcpConnection.java
public class TcpConnection {
    ......

    // Load the shared library
    static {
        System.loadLibrary("distributed_inference_demo");
    }
}
```

# Native-Lib Interface Function Documentation
1. `createSession(model_path: String): Long`. Createing Onnxruntime inference session and load model from the `model_path` String argument.

   1. Input: `model_path`(datatype `String`).
  
   2. return: java `long` value.
       
2. `releaseSession(session: Long): void`. Release Onnxruntime inference session once all inference tasks are finished. This function is defined to optimize the memory usage.

   1. Input: `session` (datatype `Long`).

3. `createHuggingFaceTokenizer(tokenizer_file_path_j:String):Long`. Creating a huggingface tokenizer for tokenizing language model string input.

   1. Input: `tokenizer_file_path_j` (datatype `String`).
  
   2. return: java `long` value.
  
4. `createSentencePieceTokenizer(tokenizer_file_path_j: String):Long`. Creating a SentencePiece tokenizer for tokenizing langauge model string input.

   1. Input: `tokenizer_file_path_j` (datatype `String`).
  
   2. return: java `long` value.

5. `performInferenceMaster(session: Long, input_ids_j:IntArray): ByteArray`. Performing inference on the header machine.

   1. Input: `session` (datatype `Long`), `input_id_j` (datatype `IntArray`)  encoded token id int array.

   2. return: serialized inference output tensor vector in bytearray (datatype `ByteArray`).

6. `performInferenceWorker(session: Long, data:ByteArray): ByteArray`. Performing inference on worker machines given serialized tensor vector in ByteArray.

   1. Input: `session` (datatype `Long`), `data` (datatype `ByteArray`).
  
   2. return: serialized inference output tensor vector in bytearray (datatype `ByteArray`).
  
7. `binaryClassify(data:ByteArray): Int`. Performing Binary Classification task given final model output in a serialized tensor vector in ByteArray fasion. Equivalent to `argmax` operation.

   1. Input: `data` (datatype `ByteArray`).

   2. return: the max tensor logit index (datatype `Int`).

8. `greedyDecoding(data:ByteArray, input_ids_j: IntArray): IntArray`. Performing Greedy Search operation to add the highest possible token id to the original token id array.
   
   1. Input:
      
      1.  `data`(datatype `ByteArray` | datashape `(batch_size, sequence_length, vocab_size)`) final generative inference probability matrix.
         
      2.  `input_ids_j` (datatype `IntArray`) encoded token id int array.
         
   2. Output: updated token id array (datatype `IntArray`).
  
10. `encodeString(data:String, tokenizer:Long):IntArray`. Performing string encoding given specific tokenizer.
    
    1. Input:
       
       1. `data` (datatype `String`). The string to be encoded.
          
       2. `tokenizer` (datatype `Long`). The tokenizer created either by `createHuggingFaceTokenizer` or `createSentencePieceTokenizer`.
          
    2. Output:
       
       1. encoded token id array in datatype `IntArray`.

11. `decodeID(data:IntArray, tokenizer:Long): String`. Performing token id array decoding to string.
    
    1. Input:
       1. `data` (token id int array `IntArray`).
       2. `tokenizer` (datatype `Long`).

    2. Output:
       1. decoded string (`String`).
         


