import torch
import torch.nn as nn
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from onnxruntime.quantization import MatMulWeight4Quantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from pathlib import Path
import os
import shutil


def onnx_quantize_model(model_path: str):
    """
    Quantize the model, so that it can be run on mobile devices with smaller memory footprint
    """
    model_path_str = "/"+os.path.join(*model_path.split("/")[:-1])
    model_path = Path(model_path)
    quantized_model_path = model_path.with_name(model_path.stem+"_quant").with_suffix(model_path.suffix)
    quantized_model_name = quantized_model_path.name.split(".")[0]
    quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QInt8, use_external_data_format=True)
    model_path.unlink()
    link_external_data(model_path_str, quantized_model_name)
    return quantized_model_path


def link_external_data(model_path: str, model_name: str):
    # Normalize the model_path to an absolute path
    model_path = os.path.abspath(model_path)
    print(f"Normalized model path: {model_path}")

    # Find the project directory by searching for "LinguaLinked"
    project_dir = model_path
    while "LinguaLinked" not in os.path.basename(project_dir) and project_dir != "/":
        project_dir = os.path.dirname(project_dir)

    if "LinguaLinked" not in os.path.basename(project_dir):
        raise RuntimeError("Could not find 'LinguaLinked' in the path hierarchy.")

    print(f"Project directory: {project_dir}")

    # Construct the original and destination paths
    original_dir = os.path.join(project_dir, model_name + ".onnx.data")
    destination_dir = os.path.join(model_path, model_name + ".onnx.data")

    print(f"Original directory: {original_dir}")
    print(f"Destination directory: {destination_dir}")

    # Check if the file exists and move it
    if os.path.exists(original_dir):
        print(f"Found external protobuf data for {model_name}")
        shutil.move(original_dir, destination_dir)
    else:
        print(original_dir)
        raise RuntimeError("External data must exist")








