from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
from transformers.utils.fx import symbolic_trace

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
saving_dir = "gptq-opt125m-4bit"
quantizer = GPTQQuantizer(bits=4,
                          dataset="wikitext2",
                          block_name_to_quantize = "model.decoder.layers",
                          model_seqlen = 2048,
                          disable_exllama=True)
quantized_model = quantizer.quantize_model(model, tokenizer)
quantizer.save(quantized_model, saving_dir)
gm = symbolic_trace(quantized_model)
print(gm.graph)
print(tokenizer.decode(quantized_model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))