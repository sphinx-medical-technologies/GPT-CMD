import torch
print("Is cuda available? ", torch.cuda.is_available())
print("What is cuDNN version:", torch.backends.cudnn.version())
print("Is cuDNN enabled?", torch.backends.cudnn.enabled)
print("Device count?", torch.cuda.device_count())
print("Current device?", torch.cuda.current_device())
print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))
x = torch.rand(5, 3)
print(x)

