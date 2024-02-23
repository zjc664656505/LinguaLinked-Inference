import errno

import numpy
import onnx
import torch
import torchvision
import os
import onnxruntime
from tqdm.auto import tqdm
import numpy as np
import json
import shutil
import zipfile
from system_pipeline.quantization.quantize import onnx_quantize_model
from system_pipeline.model_split.utils import InferenceBuffer, get_receiver_res_dependency_map, \
    get_receiver_seq_dependency_map, process_module_arrangement
import transformers
from contextlib import contextmanager
from pathlib import Path
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
import sys
import struct
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
from onnx import helper, checker
from onnx import TensorProto
import gc
import time


def flatten_list(input_list):
    output = []
    for item in input_list:
        output.extend(item)
    return output


def serialize_tensors(tensors, data_types):
    """
    Serialize a list of numpy arrays with provided ONNX tensor data types.

    Args:
    - tensors (list): List of numpy arrays.
    - data_types (list): List of ONNX tensor element data types corresponding to tensors.

    Returns:
    - bytearray: Serialized data.
    """
    assert len(tensors) == len(data_types), "Number of tensors and data types must match."

    serialized_data = []

    # Add the number of tensors
    serialized_data.append(struct.pack('<Q', len(tensors)))  # Using Q for size_t

    for tensor, dtype in zip(tensors, data_types):
        # Add tensor data type
        serialized_data.append(struct.pack('<I', dtype))  # Using I for ONNXTensorElementDataType

        # Add number of dimensions
        serialized_data.append(struct.pack('<Q', tensor.ndim))

        # Add tensor shape
        for dim in tensor.shape:
            serialized_data.append(struct.pack('<q', dim))  # Using q for int64_t

        # Add tensor data
        serialized_data.append(tensor.tobytes())

    return b''.join(serialized_data)


def get_return_names_fx(module):
    graph = module.graph
    out = []
    for node in list(graph.nodes)[::-1]:
        if node.op == "output":
            out = node.args[0]

    if isinstance(out, list):
        return [i.name for i in out]

    if isinstance(out, torch.fx.Node):
        return [out.name]


def get_placeholder_names_fx(module):
    graph = module.graph
    out = []
    for node in graph.nodes:
        if node.op == "placeholder":
            out.append(node.name)
    return out


def torch_to_onnx_residual(module_list: list,
                           input: torch.Tensor,
                           export_path: str,
                           sequential_dependency_map: dict,
                           residual_dependency_map: dict,
                           transformer_option: bool = True,
                           quantization_option: bool = True):
    """
        Method for converting a list of GraphModules to onnx models.
        arguments:
        1. module_list (List): list of torch.fx GraphModules.
        2. input (torch.Tensor): input tensor as dummy input for onnx model conversion.
        3. export_path (str): path for saving the converted onnx modules.
        4. transformer_option (bool): boolean value to indicate whether using transformers model from huggingface.
    """
    save_path = export_path
    if module_list == [] or input == None or save_path == "":
        raise RuntimeError("Given method arguments cannot be empty.")
    buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
    MODEL_INPUT_NAMES = {}
    MODEL_OUTPUT_NAMES = {}

    model_output_name = []
    model_input_name = []
    model_input = [[input]]
    print("STARTING MODEL CONVERSION: PYTORCH TO ONNX.")

    for i in tqdm(range(len(module_list))):
        assert type(module_list[i]).__qualname__.startswith("GraphModule")
        save_path = export_path
        model_input_name = get_placeholder_names_fx(module_list[i])
        model_output_name = get_return_names_fx(module_list[i])

        # replace the model output_name if the input_name is the same with the output_name
        for in_idx in range(len(model_input_name)):
            for out_idx in range(len(model_output_name)):
                if model_input_name[in_idx] == model_output_name[out_idx]:
                    model_output_name[out_idx] = f"{model_output_name[out_idx]}_1"

        module_name = {}
        module_name["input"] = model_input_name
        module_name["output"] = model_output_name
        MODEL_INPUT_NAMES[f"module_{i}"] = model_input_name
        MODEL_OUTPUT_NAMES[f"module_{i}"] = model_output_name

        save_path = f"{save_path}/module{i}/"

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        with torch.no_grad():
            # TODO: currently torch_to_onnx_residual only supports transformer types of model from huggingface
            #   further implementation can be done later.
            if transformer_option:
                if i == 0:  # handle the first submodule edge case
                    model_output = module_list[i].forward(model_input[i][0])
                    buffer.update(i, forward_output=model_output)
                elif i == len(module_list) - 1:  # handle the last submodule edge case
                    current_inputs = buffer.get_inputs_for_submodule(i)
                    print(len(current_inputs))

                    model_input.append(current_inputs)
                    model_output = module_list[i].forward(*current_inputs)
                else:
                    current_inputs = buffer.get_inputs_for_submodule(i)
                    model_input.append(current_inputs)
                    model_output = module_list[i].forward(*current_inputs)
                    buffer.update(i, model_output)

        if len(model_input[i]) == 1:
            model_input[i] = model_input[i][0]
            dynamic_axis = {}
            for idx in range(len(model_input_name)):
                dynamic_axis[model_input_name[idx]] = [j for j in range(len(model_input[idx].shape))]
            torch.onnx.export(
                module_list[i], model_input[i],
                f"{save_path}/module_{i}.onnx",
                input_names=model_input_name, output_names=model_output_name,
                dynamic_axes=dynamic_axis
            )
        else:
            dynamic_axis = {}
            assert len(model_input_name) == len(model_input[i])
            for idx in range(len(model_input_name)):
                if torch.is_tensor(model_input[i][idx]):
                    name = model_input_name[idx]
                    dynamic_axis[name] = [j for j in range(len(model_input[i][idx].shape))]
            torch.onnx.export(
                module_list[i], tuple(t for t in model_input[i]),
                f"{save_path}/module_{i}.onnx",
                input_names=model_input_name, output_names=model_output_name,
                dynamic_axes=dynamic_axis)

        print("SAVING CONFIG FILES FOR DISTRIBUTED INFERENCE...")
        with open(f"{save_path}/module_name_{i}.json", "w") as file:
            json.dump(module_name, file)
        print("DISTRIBUTED CONFIG FILES SAVING FINISHED.")

        gc.collect()

        # TODO: Should we support INT4 quantization?
        if quantization_option:
            print(f"STARTING INT8 QUANTIZATION on module_{i}.onnx")
            model_pth = f"{save_path}/module_{i}.onnx"
            onnx_quantize_model(model_pth)
            time.sleep(0.5)
            gc.collect()

    print("MODEL CONVERSION FINISHED.")
    sender_seq_dep_map = {key: {inner_key: inner_value for inner_key, inner_value in sorted(value.items())} for
                          key, value in sequential_dependency_map.items()}
    sender_res_dep_map = {key: {inner_key: inner_value for inner_key, inner_value in sorted(value.items())} for
                          key, value in residual_dependency_map.items()}
    receiver_seq_dep_map = get_receiver_seq_dependency_map(sender_seq_dep_map)
    receiver_res_dep_map = get_receiver_res_dependency_map(sender_res_dep_map)

    print("SAVING CONFIG FILES FOR LOCAL INFERENCE...")
    with open(f"{export_path}/model_output_names.json", "w") as file:
        json.dump(MODEL_OUTPUT_NAMES, file)
    with open(f"{export_path}/model_input_names.json", "w") as file:
        json.dump(MODEL_INPUT_NAMES, file)
    with open(f"{export_path}/sender_seq_dep_map.json", "w") as file:
        json.dump(sender_seq_dep_map, file)
    with open(f"{export_path}/sender_res_dep_map.json", "w") as file:
        json.dump(sender_res_dep_map, file)
    with open(f"{export_path}/receiver_seq_dep_map.json", "w") as file:
        json.dump(receiver_seq_dep_map, file)
    with open(f"{export_path}/receiver_res_dep_map.json", "w") as file:
        json.dump(receiver_res_dep_map, file)

    print("CONFIG FILES SAVING FINISHED.")


def torch_to_onnx(module_list, input, export_path, transformer_option=True, quantization_option=True):
    """
        Method for converting a list of GraphModules to onnx models.
        arguments:
        1. module_list (List): list of torch.fx GraphModules.
        2. input (torch.Tensor): input tensor as dummy input for onnx model conversion.
        3. export_path (str): path for saving the converted onnx modules.
        4. transformer_option (bool): boolean value to indicate whether using transformers model from huggingface.
    """
    save_path = export_path
    if module_list == [] or input == None or save_path == "":
        raise RuntimeError("Given method arguments cannot be empty.")

    MODEL_INPUT_NAMES = {}
    MODEL_OUTPUT_NAMES = {}

    model_output_name = []
    model_input_name = []
    model_input = [input]
    print("STARTING MODEL CONVERSION: PYTORCH TO ONNX.")

    for i in tqdm(range(len(module_list))):
        assert type(module_list[i]).__qualname__.startswith("GraphModule")
        save_path = export_path
        if i == 0:
            model_input_name = get_placeholder_names_fx(module_list[i])
            model_output_name = get_return_names_fx(module_list[i])
        else:
            model_input_name = MODEL_OUTPUT_NAMES[f"module_{i - 1}"]
            model_output_name = get_return_names_fx(module_list[i])

        # replace the model output_name if the input_name is the same with the output_name
        for in_idx in range(len(model_input_name)):
            for out_idx in range(len(model_output_name)):
                if model_input_name[in_idx] == model_output_name[out_idx]:
                    model_output_name[out_idx] = f"{model_output_name[out_idx]}_1"

        module_name = {}
        module_name["input"] = model_input_name
        module_name["output"] = model_output_name
        MODEL_INPUT_NAMES[f"module_{i}"] = model_input_name
        MODEL_OUTPUT_NAMES[f"module_{i}"] = model_output_name

        save_path = f"{save_path}/module{i}/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        print("SAVING CONFIG FILES FOR DISTRIBUTED INFERENCE...")
        with open(f"{save_path}/module_name_{i}.json", "w") as file:
            json.dump(module_name, file)
        print("DISTRIBUTED CONFIG FILES SAVING FINISHED.")

        with torch.no_grad():
            if transformer_option:
                if i == 0:
                    model_output = module_list[i](model_input[i])
                    model_input.append(model_output)
                elif 0 <= i < len(module_list) - 1:
                    model_output = module_list[i](*model_input[i])
                    model_input.append(model_output)
                else:
                    print("no action required")
            else:
                if 0 <= i < len(module_list) - 1:
                    model_output = module_list[i](model_input[i])
                    model_input.append(model_output)
                else:
                    print("no action required")

        if len(model_input[i]) == 1:
            dynamic_axis = {}
            for idx in range(len(model_input_name)):
                dynamic_axis[model_input_name[idx]] = [j for j in range(len(model_input[idx].shape))]
            torch.onnx.export(
                module_list[i], model_input[i],
                f"{save_path}/module_{i}.onnx",
                input_names=model_input_name, output_names=model_output_name,
                dynamic_axes=dynamic_axis
            )
        else:
            dynamic_axis = {}
            assert len(model_input_name) == len(model_input[i])
            for idx in range(len(model_input_name)):
                if torch.is_tensor(model_input[i][idx]):
                    name = model_input_name[idx]
                    dynamic_axis[name] = [j for j in range(len(model_input[i][idx].shape))]
            torch.onnx.export(
                module_list[i], tuple(t for t in model_input[i]),
                f"{save_path}/module_{i}.onnx",
                input_names=model_input_name, output_names=model_output_name,
                dynamic_axes=dynamic_axis)

        # TODO: quantization pipeline needs to be moved to the head, Junchen 7/19

        gc.collect()

        if quantization_option:
            print(f"STARTING INT8 QUANTIZATION on module_{i}.onnx")
            model_pth = f"{save_path}/module_{i}.onnx"
            onnx_quantize_model(model_pth)
            time.sleep(0.5)
            gc.collect()

    print("MODEL CONVERSION FINISHED.")

    print("SAVING CONFIG FILES FOR LOCAL INFERENCE...")
    with open(f"{export_path}/model_output_names.json", "w") as file:
        json.dump(MODEL_OUTPUT_NAMES, file)
    with open(f"{export_path}/model_input_names.json", "w") as file:
        json.dump(MODEL_INPUT_NAMES, file)
    print("CONFIG FILES SAVING FINISHED.")


def create_session(module_path, module_rank, quantization_option=True):
    """
        Method for conducting inference with onnx model.

        arguments:
        1. module_path (str): path for the saved module to be loaded.
        2. module_rank (int): module rank number for retrieving module input and output name
        3. quantization_option (bool): default set to True. Quantize model weight to int8

        return:
        1. onnx inference session: onnxruntime.InferenceSession.
        2. module (str): module name

    """
    if module_path == "":
        raise RuntimeError("Given method arguments cannot be empty.")

    module = f"module_{module_rank}"

    if quantization_option:
        onnx_session = onnxruntime.InferenceSession(f"{module_path}/{module}_quant.onnx",
                                                    providers=['CPUExecutionProvider'])
    else:
        onnx_session = onnxruntime.InferenceSession(f"{module_path}/{module}.onnx",
                                                    providers=['CPUExecutionProvider'])
    return onnx_session, module


def get_tensor_info(module_path, module_rank, quantization_option, tensor_type):
    if module_path == "":
        raise RuntimeError("Given method arguments cannot be empty.")

    module = f"module_{module_rank}"
    if quantization_option:
        module = onnx.load(f"{module_path}/{module}_quant.onnx")
    else:
        module = onnx.load(f"{module_path}/{module}.onnx")

    if tensor_type == "input":
        return [i for i in module.graph.input]
    elif tensor_type == "output":
        return [i for i in module.graph.output]


def create_inference_input(module_path, module_name, input):
    """
        arguments:
        1. module_path (str): path for the save module.
        2. module_name (str): module name for representing the module rank (module_0, module_1,...)
        3. input Sequence[np.array]: Sequence of input array for onnxruntime to run inference.

        returns:
        1. model_output_names (list): sequence of output name strings, required for onnxruntime inference.
        2. input_to_feed (dict): a dictionary containing the input name as key and input array as value, required for
        onnxruntime inference.
    """
    assert all(isinstance(elem, np.ndarray) for elem in input), "Element in input must be np.array type!"

    module = module_name
    input_to_session = {}
    with open(f"{module_path}/model_input_names.json", "r") as file:
        model_input_names = json.load(file)

    with open(f"{module_path}/model_output_names.json", "r") as file:
        model_output_names = json.load(file)

    model_input_name = model_input_names[module]
    model_output_name = model_output_names[module]

    assert len(input) == len(model_input_name), f"Input size:({len(input)}) must match with the size of model input " \
                                                f"names: ({len(model_input_name)})."

    for i in range(len(input)):
        input_to_session[model_input_name[i]] = input[i]

    return model_output_name, input_to_session


def create_inference_input_distributed(model_input_name, input):
    """
        arguments:
        1. model_input_name (dict): configuration dictionary containing model input and output names {"input": [...], "output":[...]}
        2. input Sequence[np.array]: Sequence of input array for onnxruntime to run inference.

        returns:
        1. model_output_names (list): sequence of output name strings, required for onnxruntime inference.
        2. input_to_feed (dict): a dictionary containing the input name as key and input array as value, required for
        onnxruntime inference.
    """
    assert all(isinstance(elem, np.ndarray) for elem in input), "Element in input must be np.array type!"

    input_to_session = {}
    input_name = model_input_name["input"]
    output_name = model_input_name["output"]

    assert len(input) == len(input_name), f"Input size:({len(input)}) must match with the size of model input " \
                                          f"names: ({len(input_name)})."

    for i in range(len(input)):
        input_to_session[input_name[i]] = input[i]

    return output_name, input_to_session


def onnx_module_loading(module_path, module_index, quantization_option):
    if quantization_option:
        return onnx.load(f"{module_path}module_{module_index}_quant.onnx")
    else:
        return onnx.load(f"{module_path}module_{module_index}.onnx")


def onnx_inference(onnx_session, output_names, input_to_session):
    """
        arguments:
        1. onnx_session (onnx.InferenceSession)
        2. output_names (list)
        3. input_to_session (dict)

        return:
        1. np.ndarray: the inference result
    """
    return onnx_session.run(output_names=output_names, input_feed=input_to_session)


def onnx_inference_distributed(inference_input, session, input_names):
    """
        method for running distributed inference via onnx
        arguments:
        1. inference_input Sequence[np.ndarray...
        2. session (onnx.InferenceSession)
        3. input_names: configuration dictionary containing model input and output names {"input": [...], "output":[...]}
    """
    output_names, input_to_session = create_inference_input_distributed(input_names, inference_input)
    inference_input = onnx_inference(session, output_names, input_to_session)
    return inference_input


def zip_file(rank: int, base_path: str = "./onnx_model/resnet", quantization_option=True) -> None:
    """
        Compress Target Split model files into zip file for tcp transmission.
    """
    rank = rank
    base_path = base_path
    folder_name = 'module'
    if quantization_option:
        file_names = [f'module_{rank}_quant.onnx', f'module_name_{rank}.json']
    else:
        file_names = [f'module_{rank}.onnx', f'module_name_{rank}.json']
    new_file_names = ['module.onnx', 'name.json']
    zip_file_name = 'module.zip'
    dest_folder_name = f'module{rank}'

    # Create the destination folder path
    dest_folder_path = os.path.join(base_path, dest_folder_name)
    folder_path = os.path.join(dest_folder_path, folder_name)

    if not os.path.exists(dest_folder_path) or not os.listdir(dest_folder_path):
        # Create the destination folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Move and rename the files
        for i, file_name in enumerate(file_names):
            old_file_path = os.path.join(base_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_names[i])
            shutil.copy(old_file_path, new_file_path)

        # Zip the folder
        zip_file_path = os.path.join(base_path, zip_file_name)
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

        shutil.move(zip_file_path, dest_folder_path)
        print(f'Files for module{rank} moved, renamed, and zipped successfully.')
    else:
        pass


def initialize_onnx_model(model_path, model_name_path, model_rank, quantization_option=True):
    """
        Method for initializing onnx model for distributed inference.
        args:
        1. model_path (str): where the model is saved, e.g. "onnx_model/bloom_560m"
        2. model_name_path (str): where the model name config file is saved, e.g. "onnx_model/bloom_560m/module_name_0.json"
        3. model rank (int): the rank number of the model, e.g. 0, 1, 2, 3, ... n.
    """

    # Build Model Session
    session, module_name = create_session(model_path, model_rank, quantization_option=quantization_option)

    # Read Input Format
    with open(f"{model_name_path}", "r") as file:
        input_names = json.load(file)
    return session, input_names


def get_model_size(model_path):
    """
        Method for initializing onnx model for distributed inference.
        args:
        1. model_path (str): where the model is saved, e.g. "onnx_model/bloom_560m"
        2. model_name_path (str): where the model name config file is saved, e.g. "onnx_model/bloom_560m/module_name_0.json"
        3. model rank (int): the rank number of the model, e.g. 0, 1, 2, 3, ... n.
    """
    def get_folder_size(folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size

    return get_folder_size(model_path)


def device_module_assignment_optimized(input_module_path: str,
                                       output_module_path: str,
                                       module_arrangement: list,
                                       quantization_option: bool,
                                       sequential_dependency_map: dict = None,
                                       residual_dependency_map: dict = None,
                                       load_balancing_option: bool = False):
    modules_on_each_device = module_arrangement
    print("START MODULE ASSIGNMENT ARRANGEMENT FOR EACH DEVICE.")
    for device_index in tqdm(range(len(modules_on_each_device))):
        device_folder = f"device{device_index}"
        output_path = os.path.join(output_module_path, device_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for module_index in modules_on_each_device[device_index]:
            if quantization_option:
                submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}_quant.onnx",
                                      load_external_data=True)
            else:
                submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}.onnx")
            device_output_path = os.path.join(output_path, f"module{module_index}")
            if not os.path.exists(device_output_path):
                os.makedirs(device_output_path)
            if load_balancing_option:
                onnx.save_model(submodule, f"{device_output_path}/module_{module_index}.onnx",
                                save_as_external_data=True)
            else:
                onnx.save_model(submodule, f"{device_output_path}/module.onnx", save_as_external_data=True)
    print("MODULE ASSIGNMENT ARRANGEMENT FOR EACH DEVICE IS FINISHED.")


def device_module_assignment(input_module_path: str,
                             output_module_path: str,
                             module_arrangement: list,
                             quantization_option: bool,
                             sequential_dependency_map: dict = None,
                             residual_dependency_map: dict = None,
                             load_balancing_option: bool = False):
    """
        This function is entriely based on the optimizer's module_arrangement, where
        the module_arrangement is a m x n dimension matrix represented as a nested list.
        Assuming now we have 3 devices, where m is 3 and n is 5.
        Then, the module_arrangement is represented as follow:
        [
            [1, 1, 0, 0 ,0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1]
        ]
        In this example, for device 1, where the row dimension m = 1, it needs submodule on index 0 and 1 to be sent
        from the server. However, since device1 and device2 are having overlapping module on column dimension 1,
        therefore, the submodule on the device1 at index0 needs to be merged and the submodule on index 1 stays dynamic.

        So, for each device, 1, 2, 3, they should have 2 lists: to_merge_index and dynamic_index as follow
        device1: to_merge_index = [0], dynamic_index = [1]
        device2: to_merge_index = [2], dynamic_index = [1,3]
        device3: to_merge_index = [4], dynamic_index = [3]
    """
    # get model index
    split_index_lst = []
    processed_module_arrangement = process_module_arrangement(module_arrangement)
    for i in range(len(processed_module_arrangement)):
        to_merge_index, dynamic_index = processed_module_arrangement[i]
        split_index_lst.append([to_merge_index, dynamic_index])

    # get total number of modules after split
    total_num_modules = len(module_arrangement[0])

    # get the module index range for each device assignment
    modules_on_each_device = []
    for device_index in range(len(module_arrangement)):
        module_assignment = []
        for module_index in range(len(module_arrangement[device_index])):
            if module_arrangement[device_index][module_index] == 1:
                module_assignment.append(module_index)
        modules_on_each_device.append(module_assignment)
    print(modules_on_each_device)

    def save_modules(module_index: int, quantization_option: bool, input_module_path: str, output_path: str):
        if quantization_option:
            submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}_quant.onnx")
        else:
            submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}.onnx")
        device_output_path = os.path.join(output_path, f"module{module_index}")
        if not os.path.exists(device_output_path):
            os.makedirs(device_output_path)
        if load_balancing_option:
            onnx.save(submodule, f"{device_output_path}/module_{module_index}.onnx")
        else:
            onnx.save(submodule, f"{device_output_path}/module.onnx")

    def load_modules(module_index: int, quantization_option: bool, input_module_path: str):
        if quantization_option:
            submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}_quant.onnx")
        else:
            submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}.onnx")
        return submodule

    def get_seq_res_names(onnx_module, module_index, seq_dep_map, res_dep_map):
        model_output_names = [o.name for o in onnx_module.graph.output]
        if module_index in seq_dep_map:
            seq_output_names = [model_output_names[i] for i in seq_dep_map[module_index][module_index + 1]]
        if module_index in res_dep_map:
            res_output_names = []
            for key, val in res_dep_map[module_index].items():
                for indice in val:
                    res_name = model_output_names[indice]
                    if res_name not in res_output_names:
                        res_output_names.append(res_name)
        else:
            res_output_names = []
        return model_output_names, seq_output_names, res_output_names

    def construct_output_names(model1_out_names, model2_out_names,
                               model1_seq_out_names, model2_seq_out_names,
                               model1_res_out_names, model2_res_out_names):
        out_names = []
        res_out_names = []
        if model1_res_out_names:
            if model2_res_out_names:
                out_names = model2_seq_out_names + model1_res_out_names + model2_res_out_names
                res_out_names = model1_res_out_names + model2_res_out_names
            else:
                out_names = model2_out_names + model1_res_out_names
                res_out_names = model1_res_out_names
        else:
            if model2_res_out_names:
                out_names = model2_seq_out_names + model2_res_out_names
                res_out_names = model2_res_out_names
            else:
                out_names = model2_out_names
                res_out_names = []
        return out_names, res_out_names

    print("START MODULE ASSIGNMENT ARRANGEMENT FOR EACH DEVICE.")
    for device_index in tqdm(range(len(modules_on_each_device))):
        device_folder = f"device{device_index}"
        output_path = os.path.join(output_module_path, device_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for module_index in modules_on_each_device[device_index]:
            if quantization_option:
                submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}_quant.onnx",
                                      load_external_data=True)
            else:
                submodule = onnx.load(f"{input_module_path}/module{module_index}/module_{module_index}.onnx")
            device_output_path = os.path.join(output_path, f"module{module_index}")
            if not os.path.exists(device_output_path):
                os.makedirs(device_output_path)
            if load_balancing_option:
                onnx.save_model(submodule, f"{device_output_path}/module_{module_index}.onnx",
                                save_as_external_data=True)
            else:
                onnx.save_model(submodule, f"{device_output_path}/module.onnx", save_as_external_data=True)
    print("MODULE ASSIGNMENT ARRANGEMENT FOR EACH DEVICE IS FINISHED.")


def model_output_size(tokenizer,
                      onnx_module_path,
                      split_size,
                      quantization_option=True,
                      residual_connection=True,
                      sequential_dependency_map=None,
                      residual_dependency_map=None):
    test_input = [tokenizer("positive " * i, return_tensors="pt")["input_ids"] for i in [8, 16, 32, 64, 128]]
    onnx_sessions = []
    module_names = []
    output_size_map = {}

    for rank in range(split_size):
        module_path = f"{onnx_module_path}/module{rank}"
        name_path = f"{onnx_module_path}/module{rank}/module_name_{rank}.json"
        session, name = initialize_onnx_model(module_path, name_path, rank, quantization_option=quantization_option)
        onnx_sessions.append(session)
        module_names.append(name)

    for input_id in test_input:
        input_ids_np = input_id.numpy()  # input_for_inference shape is (1,8)
        input_for_inference = input_ids_np.copy()
        if residual_connection:
            buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
            for index, submodule in enumerate(onnx_sessions):
                if not index in output_size_map:
                    output_size_map[index] = []
                if index == 0:  # handle the first submodule edge case
                    current_inputs = input_for_inference
                    output = onnx_inference_distributed([current_inputs], submodule, module_names[index])
                    current_output_size = []
                    for i in range(len(output)):
                        output_ele_size: numpy.ndarray.nbytes = output[i].nbytes
                        current_output_size.append(output_ele_size)
                    output_size_map[index].append(current_output_size)
                    buffer.update(index, forward_output=output)
                elif index == len(onnx_sessions) - 1:  # handle the last submodule edge case
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, module_names[index])
                    current_output_size = []
                    for i in range(len(output)):
                        output_ele_size: numpy.ndarray.nbytes = output[i].nbytes
                        current_output_size.append(output_ele_size)
                    output_size_map[index].append(current_output_size)
                else:
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, module_names[index])
                    current_output_size = []
                    for i in range(len(output)):
                        output_ele_size: numpy.ndarray.nbytes = output[i].nbytes
                        current_output_size.append(output_ele_size)
                    output_size_map[index].append(current_output_size)
                    buffer.update(index, output)
        else:
            for index, submodule in enumerate(onnx_sessions):
                if not index in output_size_map:
                    output_size_map[index] = []
                if index == 0:
                    input_for_inference = onnx_inference_distributed([input_for_inference], onnx_sessions[index],
                                                                     module_names[index])
                    current_output_size = []
                    for i in range(len(input_for_inference)):
                        output_ele_size: numpy.ndarray.nbytes = input_for_inference[i].nbytes
                        current_output_size.append(output_ele_size)
                    output_size_map[index].append(current_output_size)
                else:
                    input_for_inference = onnx_inference_distributed(input_for_inference, onnx_sessions[index],
                                                                     module_names[index])
                    current_output_size = []
                    for i in range(len(input_for_inference)):
                        output_ele_size: numpy.ndarray.nbytes = input_for_inference[i].nbytes
                        current_output_size.append(output_ele_size)
                    output_size_map[index].append(current_output_size)

    def sum_and_average(matrix):
        column_sum = [sum(col) for col in zip(*matrix)]
        column_avg = [s / len(matrix) for s in column_sum]

        return column_avg

    for key, val in output_size_map.items():
        output_size_map[key] = sum_and_average(output_size_map[key])

    if residual_connection:
        final_output_size_map = {}
        for key in output_size_map:
            if key in sequential_dependency_map:
                mem_size_arr = output_size_map[key]
                mem_index = sequential_dependency_map[key][key + 1]
                final_output_size_map[key] = {"seq": {key + 1: [mem_size_arr[i] for i in mem_index]}, "res": {}}
            if key in residual_dependency_map:
                mem_size_arr = output_size_map[key]
                for tgt_key, val in residual_dependency_map[key].items():
                    if tgt_key not in final_output_size_map[key]["res"]:
                        final_output_size_map[key]["res"][tgt_key] = [mem_size_arr[i] for i in val]
            # currently assume that the last module sents the output to the first module to simulate the distributed case
            if key == len(onnx_sessions) - 1:
                mem_size_arr = output_size_map[key]
                final_output_size_map[key] = {"seq": {0: mem_size_arr}, "res": {}}

        out_path = f"{onnx_module_path}/model_output_mem_size.json"
        with open(out_path, "w") as file:
            json.dump(final_output_size_map, file)
        return out_path, final_output_size_map
    else:
        out_path = f"{onnx_module_path}/model_output_mem_size.json"
        with open(out_path, "w") as file:
            json.dump(output_size_map, file)
        return out_path, output_size_map
