import torch
import random
from system_pipeline.model_split.model_split import *
import json
from deepspeed.profiling.flops_profiler.profiler import get_model_profile
import numpy as np


class InferenceBuffer:
    def __init__(self, sequential_dependency_map: dict, residual_dependency_map: dict):
        self.sequential_outputs = {}
        self.residual_outputs = {}
        self.sequential_dependency_map = sequential_dependency_map
        self.residual_dependency_map = residual_dependency_map

    def update(self, index, forward_output: list):
        self.sequential_outputs[index] = [forward_output[i] for i in self.sequential_dependency_map[index][index + 1]]

        if index in self.residual_dependency_map:
            self.residual_outputs[index] = {}
            for key, val in self.residual_dependency_map[index].items():
                self.residual_outputs[index][key] = [forward_output[i] for i in val]

    def get_inputs_for_submodule(self, index: int):
        inputs = []

        # Get sequential dependencies
        if index - 1 in self.sequential_outputs:
            # print(f"Sequential passing from submodule {index - 1} to submodule {index} (current).")
            inputs.extend(self.sequential_outputs[index - 1])

        # Get residual dependencies
        for src_key, src_val in self.residual_dependency_map.items():
            for tgt_key, _ in src_val.items():
                if index == tgt_key:
                    # print(f"Residual passing from submodule {src_key} to submodule {index} (current).")
                    inputs.extend(self.residual_outputs[src_key][index])

        return inputs  # Convert list to tuple to match the input format


def create_model_alloc(module_arrangmemt: np.ndarray):
    """
    Function to split model based on optimized module arragement to each of the devices For example: Given module
    arrangement input: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1]]

    This function returns:
    1. moduel_split_index: [[0, 8], [8, 9], [9, 44], [44, 45], [45, 50]]
    2. device_module_allocation_map: {0: {'fixed': [0], 'dynamic': [1]}, 1: {'fixed': [2], 'dynamic': [3]}, 2: {'fixed': [4], 'dynamic': [3]}}
    """

    module_arrangmemt:list = module_arrangmemt.tolist()
    print(module_arrangmemt)

    # create map to map module arrang list to the device id
    if not module_arrangmemt:
        raise RuntimeError("module_arrangment cannot be empty!")
    module_device_map = {}
    for i in range(len(module_arrangmemt)):
        module_device_map[tuple(module_arrangmemt[i])] = i

    print(f"\nModule-device Map:\n{module_device_map}")
    def index_of_first_one(lst):
        try:
            return lst.index(1)
        except ValueError:
            return float('inf')

    # sort module arrangement
    module_arrangmemt = sorted(module_arrangmemt, key=index_of_first_one)

    model_alloc = []
    device_model_alloc_map = {}

    # initialize the device-model allocation map
    for device_idx in range(len(module_arrangmemt)):
        device_model_alloc_map[device_idx] = {"fixed": [], "dynamic": []}

    for device_idx in range(len(module_arrangmemt)):
        temp_alloc_fixed = []
        temp_alloc_dynamic = []
        module_idxs = tuple(module_arrangmemt[device_idx])
        device_id = module_device_map[module_idxs]

        if device_idx + 1 == len(module_arrangmemt):
            temp_alloc_fixed = [model_alloc[-1][-1], len(module_arrangmemt[device_idx])]
            model_alloc.append(temp_alloc_fixed)
            device_model_alloc_map[device_id]["fixed"].append(model_alloc.index(temp_alloc_fixed))
            break

        device_alloc = module_arrangmemt[device_idx]
        next_device_alloc = module_arrangmemt[device_idx + 1]

        for module_idx in range(len(device_alloc)):
            if device_alloc[module_idx] == 1 and (device_alloc[module_idx] != next_device_alloc[module_idx]):
                temp_alloc_fixed.append(module_idx)
            elif device_alloc[module_idx] == 1 and (device_alloc[module_idx] == next_device_alloc[module_idx]):
                temp_alloc_dynamic.append(module_idx)

        # reorganize the temp alloc fixed to range format
        if temp_alloc_fixed:
            temp_alloc_fixed.append(temp_alloc_fixed[-1] + 1)
            if not model_alloc:
                temp_alloc_fixed = [temp_alloc_fixed[0], temp_alloc_fixed[-1]]
            else:
                temp_alloc_fixed = [model_alloc[-1][-1], temp_alloc_fixed[-1]]
            model_alloc.append(temp_alloc_fixed)
            device_model_alloc_map[device_id]["fixed"].append(model_alloc.index(temp_alloc_fixed))

        # temp alloc dynamic could be empty
        if temp_alloc_dynamic:
            temp_alloc_dynamic.append(temp_alloc_dynamic[-1] + 1)
            for i in range(temp_alloc_dynamic[0], temp_alloc_dynamic[-1]):
                dynamic_module_index = [i, i+1]
                model_alloc.append(dynamic_module_index)
                module_idxs_current = tuple(module_arrangmemt[device_idx])
                module_idxs_next = tuple(module_arrangmemt[device_idx+1])
                device_id_current = module_device_map[module_idxs_current]
                device_id_next = module_device_map[module_idxs_next]
                device_model_alloc_map[device_id_current]["dynamic"].append(model_alloc.index(dynamic_module_index))
                device_model_alloc_map[device_id_next]["dynamic"].append(model_alloc.index(dynamic_module_index))

        # check whether current device holds all the modules
        if model_alloc[-1][-1] == len(device_alloc):
            break

    print(f"model allocation: {model_alloc}")
    print(f"device-model map: {device_model_alloc_map}")
    return model_alloc, device_model_alloc_map


def model_sharding(model,
                   split_size,
                   remote_option=False,
                   transformer_model_option=False,
                   residual_connection=True,
                   debug_mod=False,
                   model_allocation=[],
                   split_option="fixed"):
    split_size = split_size

    # run split module
    split = ModelSplit(model, debug_mod, model_allocation=model_allocation, split_option=split_option)
    if residual_connection:
        modules, sequential_dependency_map, residual_dependency_map = split.split_module(split_size,
                                                                                         remote_option=remote_option,
                                                                                         transformer_model_option=transformer_model_option,
                                                                                         residual_connection=residual_connection)
        return modules, sequential_dependency_map, residual_dependency_map, split.max_split_size
    else:
        modules = split.split_module(split_size,
                                     remote_option=remote_option,
                                     transformer_model_option=transformer_model_option,
                                     residual_connection=False
                                     )
        return modules, split.max_split_size


def get_receiver_seq_dependency_map(seq_dependency_map: dict):
    # Increment outer keys
    new_data = {k + 1: v for k, v in seq_dependency_map.items()}

    # Increment inner keys
    for key, value in new_data.items():
        new_data[key] = {k - 1: v for k, v in value.items()}

    return new_data


def get_receiver_res_dependency_map(res_dependency_map: dict):
    new_data = {}
    for outer_key, inner_dict in res_dependency_map.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in new_data:
                new_data[inner_key] = {}
            new_data[inner_key][outer_key] = value
    return new_data


def generate_fake_module_arrangement(rows, cols):
    matrix = []

    # Start for the first row
    number_of_ones_first_row = random.randint(1, cols - (rows - 1))  # Ensure space for the following rows
    current_row = [1] * number_of_ones_first_row + [0] * (cols - number_of_ones_first_row)
    matrix.append(current_row)

    for i in range(1, rows):
        # Find the index of the second last occurrence of 1
        indices_of_ones = [index for index, x in enumerate(current_row) if x == 1]

        # If there's only one '1', take its index. Otherwise, take the second last
        start_index = indices_of_ones[-2] if len(indices_of_ones) > 1 else indices_of_ones[0]

        # If we're on the last row, fill up with 1s after the first available position
        if i == rows - 1:
            new_row = [0] * (start_index + 1) + [1] * (cols - start_index - 1)
        else:
            # Calculate how many 1s we can have for this row
            remaining_cols = cols - start_index - 1
            number_of_ones = random.randint(1, remaining_cols - 1)  # Leave space for the next rows

            # Construct next row
            new_row = [0] * (start_index + 1)
            new_row += [1] * number_of_ones
            new_row += [0] * (cols - len(new_row))

        matrix.append(new_row)
        current_row = new_row

    return matrix


def process_module_arrangement(matrix):
    devices = len(matrix)
    modules = len(matrix[0])

    # Initialize result
    result = []

    for i in range(devices):
        to_merge_index = []
        dynamic_index = []

        for j in range(modules):
            devices_needing_module = [device for device, val in enumerate(matrix) if val[j] == 1]
            if len(devices_needing_module) > 1 and i in devices_needing_module:
                dynamic_index.append(j)
            elif len(devices_needing_module) == 1 and i == devices_needing_module[0]:
                to_merge_index.append(j)

        result.append([to_merge_index, dynamic_index])

    return result


def get_module_flops(modules, tokenizer, sequential_dependency_map=None, residual_dependency_map=None):
    # TODO: This function might need to be modified since it's currently averaging the model flops based on input_id
    #  length which could be inaccurate
    input_for_export = \
    tokenizer("University of California Irvine is a public university located in", return_tensors="pt")["input_ids"]
    module_flop_map = {}

    if sequential_dependency_map and residual_dependency_map:
        buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
        for index, submodule in enumerate(modules):
            # Special handling for submodule 0
            if index == 0:  # handle the first submodule edge case
                current_inputs = input_for_export
                flop, _, _ = get_model_profile(submodule, args=[current_inputs], print_profile=False)
                module_flop_map[index] = flop
                output = submodule.forward(current_inputs)
                buffer.update(index, forward_output=output)
            elif index == len(modules) - 1:  # handle the last submodule edge case
                current_inputs = buffer.get_inputs_for_submodule(index)
                flop, _, _ = get_model_profile(submodule, args=current_inputs, print_profile=False)
                module_flop_map[index] = flop
                output = submodule.forward(*current_inputs)
            else:
                current_inputs = buffer.get_inputs_for_submodule(index)
                flop, _, _ = get_model_profile(submodule, args=current_inputs, print_profile=False)
                module_flop_map[index] = flop
                output = submodule.forward(*current_inputs)
                buffer.update(index, output)
    else:
        output = None
        for index, submodule in enumerate(modules):
            if index == 0:
                current_inputs = input_for_export
                flop, _, _ = get_model_profile(submodule, args=[current_inputs], print_profile=False)
                module_flop_map[index] = flop
                output = submodule.forward(current_inputs)
            else:
                current_inputs = output
                flop, _, _ = get_model_profile(submodule, args=current_inputs, print_profile=False)
                module_flop_map[index] = flop
                output = submodule.forward(*current_inputs)

    def flops_from_string(s):
        scale = {
            'K': 10 ** 3,  # Kilo
            'M': 10 ** 6,  # Mega
            'G': 10 ** 9,  # Giga
            'T': 10 ** 12,  # Tera
            'P': 10 ** 15,  # Peta
            'E': 10 ** 18,  # Exa
            'Z': 10 ** 21,  # Zetta
            'Y': 10 ** 24  # Yotta
        }

        for key, value in scale.items():
            if key in s:
                num = float(s.replace(key, '').strip())  # Remove the scale character and strip whitespace
                return num * value

        return float(s)  # If no scale character is found

    # average the flops for each module
    for key, val in module_flop_map.items():
        module_flop_map[key] = int(flops_from_string(val))

    return module_flop_map


def max_split_size(model, transformer_model_option=False):
    split = ModelSplit(model)
    model_graph = split.get_model_graph(transformer_model_option=transformer_model_option)
    return split.get_max_split(model_graph)