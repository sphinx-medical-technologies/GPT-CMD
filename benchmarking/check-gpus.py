import torch
import subprocess
import psutil

print("Is cuda available? ", torch.cuda.is_available())
print("What is cuDNN version:", torch.backends.cudnn.version())
print("Is cuDNN enabled?", torch.backends.cudnn.enabled)
print("Device count?", torch.cuda.device_count())
print("Current device?", torch.cuda.current_device())
print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))
x = torch.rand(5, 3)
print(x)

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

