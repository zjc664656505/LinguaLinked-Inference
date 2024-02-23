from system_pipeline.model_split.model_split import ModelSplit
from system_pipeline.model_split.utils import model_sharding, max_split_size, InferenceBuffer, \
    get_receiver_seq_dependency_map, get_receiver_res_dependency_map, process_module_arrangement, \
    generate_fake_module_arrangement, get_module_flops
import torch
import numpy as np
import random
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
import time
from system_pipeline.onnx_backend.onnx import torch_to_onnx_residual, model_output_size, device_module_assignment, torch_to_onnx
from util.model_card import ModelCard
from system_pipeline.onnx_backend.optimization import Optimizer
import onnx
from SecureConnection.monitor import Monitor
import threading

def run_residual_inference(input_for_export, model, split_size):
    # split model based on max number model split available
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)

    # try sample inference with buffer mechanism
    buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
    begin_time = time.time()
    for index, submodule in enumerate(modules):
        # Special handling for submodule 0
        print(f"\nResidual inference on submodule {index}.")
        if index == 0:  # handle the first submodule edge case
            current_inputs = input_for_export
            output = submodule.forward(current_inputs)
            buffer.update(index, forward_output=output)
        elif index == len(modules) - 1:  # handle the last submodule edge case
            current_inputs = buffer.get_inputs_for_submodule(index)
            output = submodule.forward(*current_inputs)
        else:
            current_inputs = buffer.get_inputs_for_submodule(index)
            for tensors in current_inputs:
                print(tensors.dtype)
            output = submodule.forward(*current_inputs)
            buffer.update(index, output)
    end_time = time.time()
    print(f"Residual inference time: {end_time - begin_time}")


def run_sequential_inference(input_for_export, model):
    split_size = max_split_size(model, transformer_model_option=True)
    modules = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False,
                             debug_mod=False)
    output = input_for_export
    begin_time = time.time()
    for idx in range(len(modules)):
        if idx == 0:
            output = modules[idx](output)
        else:
            output = modules[idx](*output)
    end_time = time.time()
    print(f"\nSequential inference time: {end_time - begin_time}")


def run_residual_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    torch_to_onnx_residual(modules, input_for_export, export_path, sequential_dependency_map, residual_dependency_map,
                           quantization_option=quantization_option)


def run_sequential_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False)
    torch_to_onnx(modules, input_for_export, export_path,
                           quantization_option=quantization_option)


def run_model_output_size(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)

    model_output_size(tokenizer,
                      "/Users/junchenzhao/LinguaLinked/onnx_model",
                      split_size,
                      quantization_option=True,
                      residual_connection=True,
                      sequential_dependency_map=sequential_dependency_map,
                      residual_dependency_map=residual_dependency_map)


def run_module_flop_test(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    get_module_flops(modules, tokenizer, sequential_dependency_map, residual_dependency_map)


if __name__ == "__main__":
    # model_name = "bigscience/bloom-560m"
    # torch.fx.wrap('len')
    #
    # Setup seed
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    #
    # # # load model to cpu
    # model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # split_size = max_split_size(model, transformer_model_option=True)
    #
    # # test module_flops
    # run_module_flop_test(model, tokenizer, split_size)
    #
    # input_for_export = tokenizer("I love distributed machine learning!", return_tensors="pt")["input_ids"]

    # modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, 3,
    #                                                                              transformer_model_option=True,
    #                                                                              residual_connection=True,
    #                                                                              debug_mod=True)
    # run_residual_inference(input_for_export, model, 3)
    # run_sequential_inference(input_for_export, model)
    #
    # run_sequential_onnx_export(input_for_export, model, "./onnx_model/residual_test_model", split_size=6,
    #                            quantization_option=False)

    # run_residual_onnx_export(input_for_export, model, "./onnx_model/residual_module_test", split_size=3,
    #                        quantization_option=True)


    # run_model_output_size(model, tokenizer, split_size=split_size)

    # test model arrangement and model merge
    # random.seed(1)
    # module_arrangement = generate_fake_module_arrangement(3, 50)
    # print("Fake Module Arrangement Matrix:\n")
    # for row in module_arrangement:
    #     print(row)
    #
    # arrangement_result = process_module_arrangement(module_arrangement)
    # for i in range(len(arrangement_result)):
    #     print(f"Device{i} - to_merge_index: {arrangement_result[i][0]}, dynamic_index: {arrangement_result[i][1]}\n")

    modelcard = ModelCard("vicuna7b", quantization_option=False)
    #modelcard = ModelCard("bloom560m", quantization_option=False)
    mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules \
        = modelcard.prepare_optimization_info()

    ###### INFO REQUIRED FROM DEVICES TO SERVER (Monitor Part) ######
    # monitor = Monitor(15, "34567", 2)
    # thread = threading.Thread(target=monitor.start)
    # thread.start()
    #
    # monitor.is_monitor_ready.wait()
    #
    # num_devices, ping_latency, bandwidths, TotalMem, AvailMem, flop_speed = monitor.get_monitor_info()
    #
    # mem_threshold = 0.1
    # TotalMem = [m * mem_threshold for m in TotalMem]
    # AvailMem = [m * mem_threshold for m in AvailMem]

    # print("-----------------Test Optimizer Function----------------------")
    # # print(bandwidths)
    # # print(type(bandwidths))
    #
    # num_devices = 3
    # ping_latency = np.array([[float("inf"), 91.865 / 1000, 90 / 1000],
    #                          [89.33 / 1000, float("inf"), 88 / 1000],
    #                          [85.2033 / 1000, 86.33 / 1000, float("inf")]])
    # bandwidths = np.array([[float("inf"), 12.1227, 10.303],
    #                        [13.48, float("inf"), 14.1],
    #                        [11.202, 8.98, float("inf")]])
    # TotalMem = np.array([10 * 1024, 8 * 1024, 8 * 1024])
    # AvailMem = np.array([4 * 1024, 1 * 1024, 3 * 1024])
    # flop_speed = [3.31e10, 5.35e10, 5.31e10]
    #
    # ###### INFO REQUIRED FROM DEVICES TO SERVER ######
    #
    # load_balancer = Optimizer(num_devices=num_devices, num_modules=50)
    # load_balancer.process_initial_info(num_flop=module_flop_map,
    #                                    flop_speed=flop_speed,
    #                                    ping_latency=ping_latency,
    #                                    bandwidths=bandwidths,
    #                                    m2m=out_size_map,
    #                                    model_size=mem_util,
    #                                    total_mem=TotalMem,
    #                                    ava_mem=AvailMem)
    # initial_module_arrangement = load_balancer.initial_module_arrangement()
    # overlapping_module_arrangement = load_balancer.dynamic_module_arrangement()
    #
    # print("initial_module_arrangement")
    # print(initial_module_arrangement)
    # print("overlapping_module_arrangement")
    # print(overlapping_module_arrangement)
    #
    #
    # mod_out_dir = modelcard.prepare_model_to_send(module_arrangement=initial_module_arrangement)
    #
    #






