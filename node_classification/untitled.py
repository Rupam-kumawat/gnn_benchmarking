import torch
import pyg_lib  # or simply import pyg_lib if installed

print("PyTorch version:", torch.__version__)
print("CUDA version torch built with:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))