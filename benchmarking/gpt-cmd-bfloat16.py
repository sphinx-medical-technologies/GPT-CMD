from transformers import GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import time
import subprocess
import psutil

def generate_test():
	prompt="<START TRANSCRIPT>my birth date is august 8, 1942.<END TRANSCRIPT><START BIRTHDATE>"
	print("prompt [" + str(prompt) + "]")
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
		attention_mask=tokens["attention_mask"],
        	num_beams=5,
		use_cache=False)

	text = tokenizer.decode(output[0])
	print(text)

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
print("Loaded tokenizer [" + str(tokenizer) + "]")
model = AutoModelForCausalLM.from_pretrained(model_id,device_map={"":0},torch_dtype=torch.bfloat16)
print("Loaded model [" + str(model) + "]")
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

benchmark_function(generate_test)

