from transformers import GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import time
import subprocess
import psutil

model_id = "/home/silvacarl/Desktop/models/gpt-cmd"
print(model_id)

print("Loading GPT-CMD...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map={"":0},torch_dtype=torch.float16)
end_time = time.time() - start_time
print("Model load time: ", end_time)

def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert to GB
    gpu_memory = int(result) / 1024
    return gpu_memory

# Print GPU memory usage in GB
print('GPU Memory usage:', get_gpu_memory(), 'GB')

def get_cpu_memory():
    mem = psutil.virtual_memory()
    # Convert to GB
    cpu_memory = mem.used / (1024 ** 3)
    return cpu_memory

# Print CPU memory usage in GB
print('CPU Memory usage:', get_cpu_memory(), 'GB')

prompt = "<START TRANSCRIPT>I was born on sept 18th, 1963.<END TRANSCRIPT><START BIRTHDATE>"

start = time.time()
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"]

output = model.generate(input_ids,
		temperature=0.8,
		top_k=50,
		top_p=1.0,
		min_new_tokens=0,
		max_new_tokens=30,
        	do_sample=True,
        	pad_token_id=0,
		repetition_penalty=1,
        	num_beams=5,
		use_cache=False)

text = tokenizer.decode(output[0])

end_time = time.time() - start_time
print("Generation time: ", end_time)
print(text)

