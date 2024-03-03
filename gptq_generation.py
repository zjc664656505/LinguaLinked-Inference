from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
from transformers.utils.fx import symbolic_trace
from accelerate import init_empty_weights

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
saving_dir = "gptq-opt125m-8bit"

with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
empty_model.tie_weights()
quantized_model = load_quantized_model(empty_model, save_folder=saving_dir).to("cpu")
print(tokenizer.decode(quantized_model.generate(**tokenizer("Irvine is a city that", return_tensors="pt", max_length=50).to("cpu"))[0]))