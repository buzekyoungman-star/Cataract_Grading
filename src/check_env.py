import sys
import torch
import torchvision
import cv2

print("Python路径:", sys.executable)
print("PyTorch版本:", torch.__version__)
print("Torchvision版本:", torchvision.__version__)
print("OpenCV版本:", cv2.__version__)
print("CUDA是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("当前使用CPU")