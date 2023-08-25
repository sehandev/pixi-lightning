import torch

x = torch.rand(5, 3)
print(x)

is_gpu_ok = torch.cuda.is_available()
print(is_gpu_ok)
