from transformers import GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import time
from transformers import pipeline
import subprocess
import psutil

def generate_test():
	prompt="<START TRANSCRIPT>my birth date is august 8, 1942.<END TRANSCRIPT><START BIRTHDATE>"
	print("prompt [" + str(prompt) + "]")
	text = pipe(prompt)
	print("text [" + str(text) + "]")

def benchmark_function(func, num_iterations=20):
    total_time = 0
    for i in range(num_iterations):
        start_time = time.time()  # get the current time before calling the function
        func()  # call the function
        end_time = time.time()  # get the current time after calling the function

        execution_time = end_time - start_time  # calculate the execution time
        total_time += execution_time

        print(f"Iteration {i+1}, Timestamp: {end_time}, Execution Time: {execution_time}")

    average_time = total_time / num_iterations
    print(f"Average Execution Time: {average_time}")

model_id = "/home/silvacarl/Desktop/models/gpt-cmd"
print(model_id)

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map={"":0},torch_dtype=torch.float16,load_in_8bit=True)
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
    cpu_memory = mem.used / (1024 * 1024 * 1024)
    return cpu_memory

# Print CPU memory usage in GB
print('CPU Memory usage:', get_cpu_memory(), 'GB')

pipe = pipeline('text-generation', 
                 model = model_id,
                 tokenizer = tokenizer,
                 temperature=0.8,
                 top_k=50,
                 top_p=1.0,
                 min_length=0, 
                 max_length=35, 
                 do_sample=True,
                 pad_token_id=0,
                 repetition_penalty=1,
                 num_beams=1)

benchmark_function(generate_test)

