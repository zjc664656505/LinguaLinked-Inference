//
// Created by Junchen Zhao on 7/9/23.
//

#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>

namespace utils {

    std::vector<char> SerializeInt(int value) {
        std::vector<char> byteArray(sizeof(int));
        std::memcpy(byteArray.data(), &value, sizeof(int));
        return byteArray;
    }

    int DeserializeInt(const std::vector<char>& byteArray) {
        if (byteArray.size() != sizeof(int)) {
            throw std::invalid_argument("Invalid byte array size for deserialization");
        }

        int value;
        std::memcpy(&value, byteArray.data(), sizeof(int));
        return value;
    }

    std::string JString2String(JNIEnv *env, jstring jStr) {
            if (!jStr)
                    return std::string();

            const jclass stringClass = env->GetObjectClass(jStr);
            const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes",
                                                        "(Ljava/lang/String;)[B");
            const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                               env->NewStringUTF(
                                                                                       "UTF-8"));

            size_t length = (size_t) env->GetArrayLength(stringJbytes);
            jbyte *pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

            std::string ret = std::string((char *) pBytes, length);
            env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

            env->DeleteLocalRef(stringJbytes);
            env->DeleteLocalRef(stringClass);
            return ret;
    }

    Ort::Value CopyOrtValue(const Ort::Value& original) {
        if (!original.IsTensor()) {
            throw std::runtime_error("Only tensor copying is supported in this example");
        }

        auto tensorType = original.GetTensorTypeAndShapeInfo();
        auto tensorShape = tensorType.GetShape();
        auto dataType = tensorType.GetElementType();

        // Create a new tensor with the same shape and type
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Value copiedValue = Ort::Value::CreateTensor<float>(allocator, tensorShape.data(), tensorShape.size());;


        switch(dataType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                copiedValue = Ort::Value::CreateTensor<float>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<float>(), original.GetTensorData<float>(), tensorType.GetElementCount() * sizeof(float));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                copiedValue = Ort::Value::CreateTensor<int32_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<int32_t>(), original.GetTensorData<int32_t>(), tensorType.GetElementCount() * sizeof(int32_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                copiedValue = Ort::Value::CreateTensor<int8_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<int8_t>(), original.GetTensorData<int8_t>(), tensorType.GetElementCount() * sizeof(int8_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                copiedValue = Ort::Value::CreateTensor<uint8_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<uint8_t>(), original.GetTensorData<uint8_t>(), tensorType.GetElementCount() * sizeof(uint8_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                copiedValue = Ort::Value::CreateTensor<uint16_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<uint16_t>(), original.GetTensorData<uint16_t>(), tensorType.GetElementCount() * sizeof(uint16_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                copiedValue = Ort::Value::CreateTensor<int16_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<int16_t>(), original.GetTensorData<int16_t>(), tensorType.GetElementCount() * sizeof(int16_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                copiedValue = Ort::Value::CreateTensor<int64_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<int64_t>(), original.GetTensorData<int64_t>(), tensorType.GetElementCount() * sizeof(int64_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                copiedValue = Ort::Value::CreateTensor<bool>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<bool>(), original.GetTensorData<bool>(), tensorType.GetElementCount() * sizeof(bool));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                copiedValue = Ort::Value::CreateTensor<double>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<double>(), original.GetTensorData<double>(), tensorType.GetElementCount() * sizeof(double));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                copiedValue = Ort::Value::CreateTensor<uint32_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<uint32_t>(), original.GetTensorData<uint32_t>(), tensorType.GetElementCount() * sizeof(uint32_t));
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                copiedValue = Ort::Value::CreateTensor<uint64_t>(allocator, tensorShape.data(), tensorShape.size());
                std::memcpy(copiedValue.GetTensorMutableData<uint64_t>(), original.GetTensorData<uint64_t>(), tensorType.GetElementCount() * sizeof(uint64_t));
                break;
            default:
                throw std::runtime_error("Unsupported tensor data type in this example");
        }

        return copiedValue;
    }

    std::vector<Ort::Value> CopyOrtValuesVector(const std::vector<Ort::Value>& originalVector) {
        std::vector<Ort::Value> copiedVector;
        for (const auto& value : originalVector) {
            copiedVector.push_back(CopyOrtValue(value));
        }
        return copiedVector;
    }

    // Serialize Tensor Vector to Byte Array for TCP transmission
    std::vector<char> SerializeTensorVectorToBytes(const std::vector<Ort::Value>& tensors) {
        std::vector<char> bytes;

        size_t numTensors = tensors.size();
        const char* dataPtr = reinterpret_cast<const char*>(&numTensors);
        bytes.insert(bytes.end(), dataPtr, dataPtr + sizeof(size_t));

        for (const auto& tensor : tensors) {
            if (!tensor.IsTensor()) {
                std::cerr << "Skipping non-tensor Ort::Value." << std::endl;
                continue;
            }

            Ort::TensorTypeAndShapeInfo info = tensor.GetTensorTypeAndShapeInfo();
            size_t elementCount = info.GetElementCount();

            // Record the current size of bytes to calculate the size of the added data
            size_t initialSize = bytes.size();

            // Write the tensor type to the bytes
            ONNXTensorElementDataType tensorType = info.GetElementType();
            const char* tensorTypePtr = reinterpret_cast<const char*>(&tensorType);
            bytes.insert(bytes.end(), tensorTypePtr, tensorTypePtr + sizeof(ONNXTensorElementDataType));


            // Get the shape of the tensor
            std::vector<int64_t> shape = info.GetShape();
            size_t numDimensions = shape.size();


            // Write the number of dimensions to the bytes
            const char* numDimensionsPtr = reinterpret_cast<const char*>(&numDimensions);
            bytes.insert(bytes.end(), numDimensionsPtr, numDimensionsPtr + sizeof(size_t));

            // Write each dimension to the bytes
            for (int64_t dimension : shape) {
                const char* dimensionPtr = reinterpret_cast<const char*>(&dimension);
                bytes.insert(bytes.end(), dimensionPtr, dimensionPtr + sizeof(int64_t));
            }

            size_t elementSize;
            // Write the tensor data to the bytes
            switch (tensorType) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    const float* tensorData = tensor.GetTensorData<float>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(float));
                    elementSize = sizeof(float);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                    const int8_t* tensorData = tensor.GetTensorData<int8_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(int8_t));
                    elementSize = sizeof(int8_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                    const uint8_t* tensorData = tensor.GetTensorData<uint8_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(uint8_t));
                    elementSize = sizeof(uint8_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
                    const uint16_t* tensorData = tensor.GetTensorData<uint16_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(uint16_t));
                    elementSize = sizeof(uint16_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
                    const int16_t* tensorData = tensor.GetTensorData<int16_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(int16_t));
                    elementSize = sizeof(int16_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    const int32_t* tensorData = tensor.GetTensorData<int32_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(int32_t));
                    elementSize = sizeof(int32_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    const int64_t* tensorData = tensor.GetTensorData<int64_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(int64_t));
                    elementSize = sizeof(int64_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                    const bool* tensorData = tensor.GetTensorData<bool>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(bool));
                    elementSize = sizeof(bool);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                    const double* tensorData = tensor.GetTensorData<double>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(double));
                    elementSize = sizeof(double);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
                    const uint32_t* tensorData = tensor.GetTensorData<uint32_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(uint32_t));
                    elementSize = sizeof(uint32_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
                    const uint64_t* tensorData = tensor.GetTensorData<uint64_t>();
                    const char* tensorDataPtr = reinterpret_cast<const char*>(tensorData);
                    bytes.insert(bytes.end(), tensorDataPtr, tensorDataPtr + elementCount * sizeof(uint64_t));
                    elementSize = sizeof(uint64_t);
                    break;
                }
                default:
                    std::cerr << "Unsupported tensor type for serialization: " << tensorType << std::endl;
                    break;
            }


            // Calculate the expected size
            size_t expectedSize = sizeof(ONNXTensorElementDataType) // size of the tensor type
                                  + sizeof(size_t) // size of the number of dimensions
                                  + (sizeof(int64_t) * numDimensions) // size of the tensor shape
                                  + (elementSize * elementCount); // size of the tensor data

            // Verify the total size of the serialized data for each tensor
            size_t actualSize = bytes.size() - initialSize;  // size of the added data for the current tensor
            if (actualSize != expectedSize) {
                std::cerr << "Error: Serialized tensor size (" << actualSize
                          << ") does not match expected size (" << expectedSize << ")." << std::endl;
            }
        }
        return bytes;
    }

    std::vector<Ort::Value> DeserializeTensorVectorFromBytes(const std::vector<char>& bytes) {
        std::vector<Ort::Value> tensors;

        const char* dataPtr = bytes.data();

        size_t numTensors = *reinterpret_cast<const size_t*>(dataPtr);
        dataPtr += sizeof(size_t);

        for (size_t i = 0; i < numTensors; ++i) {
            ONNXTensorElementDataType tensorType = *reinterpret_cast<const ONNXTensorElementDataType*>(dataPtr);
            dataPtr += sizeof(ONNXTensorElementDataType);

            size_t numDimensions = *reinterpret_cast<const size_t*>(dataPtr);
            dataPtr += sizeof(size_t);

            std::vector<int64_t> shape(numDimensions);
            size_t elementCount = 1;
            for (size_t j = 0; j < numDimensions; ++j) {
                shape[j] = *reinterpret_cast<const int64_t*>(dataPtr);
                dataPtr += sizeof(int64_t);
                elementCount *= shape[j];
            }

            Ort::AllocatorWithDefaultOptions allocator;
            Ort::Value tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), numDimensions);

            switch (tensorType) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<float>(), dataPtr, elementCount * sizeof(float));
                    dataPtr += elementCount * sizeof(float);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                    tensor = Ort::Value::CreateTensor<int8_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<int8_t>(), dataPtr, elementCount * sizeof(int8_t));
                    dataPtr += elementCount * sizeof(int8_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                    tensor = Ort::Value::CreateTensor<uint8_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<uint8_t>(), dataPtr, elementCount * sizeof(uint8_t));
                    dataPtr += elementCount * sizeof(uint8_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
                    tensor = Ort::Value::CreateTensor<uint16_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<uint16_t>(), dataPtr, elementCount * sizeof(uint16_t));
                    dataPtr += elementCount * sizeof(uint16_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
                    tensor = Ort::Value::CreateTensor<int16_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<int16_t>(), dataPtr, elementCount * sizeof(int16_t));
                    dataPtr += elementCount * sizeof(int16_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    tensor = Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<int32_t>(), dataPtr, elementCount * sizeof(int32_t));
                    dataPtr += elementCount * sizeof(int32_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    tensor = Ort::Value::CreateTensor<int64_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<int64_t>(), dataPtr, elementCount * sizeof(int64_t));
                    dataPtr += elementCount * sizeof(int64_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                    tensor = Ort::Value::CreateTensor<bool>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<bool>(), dataPtr, elementCount * sizeof(bool));
                    dataPtr += elementCount * sizeof(bool);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                    tensor = Ort::Value::CreateTensor<double>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<double>(), dataPtr, elementCount * sizeof(double));
                    dataPtr += elementCount * sizeof(double);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
                    tensor = Ort::Value::CreateTensor<uint32_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<uint32_t>(), dataPtr, elementCount * sizeof(uint32_t));
                    dataPtr += elementCount * sizeof(uint32_t);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
                    tensor = Ort::Value::CreateTensor<uint64_t>(allocator, shape.data(), numDimensions);
                    std::memcpy(tensor.GetTensorMutableData<uint64_t>(), dataPtr, elementCount * sizeof(uint64_t));
                    dataPtr += elementCount * sizeof(uint64_t);
                    break;
                }
                default:
                    std::cerr << "Unsupported tensor type for deserialization: " << tensorType << std::endl;
                    break;
            }

            tensors.push_back(std::move(tensor));
        }

        return tensors;
    }

} // utils