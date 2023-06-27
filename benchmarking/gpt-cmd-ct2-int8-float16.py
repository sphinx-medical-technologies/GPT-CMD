import ctranslate2
import transformers
import time
import subprocess
import psutil

def generate_test():
	prompt="<START TRANSCRIPT>my birth date is august 8, 1942.<END TRANSCRIPT><START BIRTHDATE>"
	print("prompt [" + str(prompt) + "]")

	tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
	# print(tokens)
	results = generator.generate_batch([tokens], 
		sampling_temperature=0.1, 
		sampling_topk=10, 
		sampling_topp=1, 
	        min_length=0, 
	        max_length=35, 
		repetition_penalty=1, 
		include_prompt_in_result=False, 
		beam_size=1)
#	print("results [" + str(results) + "]")
	text = tokenizer.decode(results[0].sequences_ids[0])
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

# load on GPU
generator = ctranslate2.Generator("/home/silvacarl/Desktop/models/gpt-cmd-int8-float16/",device="cuda")

# load on CPU
# generator = ctranslate2.Generator("../models/gpt-cmd-int8-float16/")

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print("tokenizer [" + str(tokenizer) + "]")

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

