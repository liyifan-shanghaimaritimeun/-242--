"""打印计算机视觉实验环境的关键版本与 CUDA / cuDNN 可用性。"""
import sys

import cv2
import numpy as np
import torch
import torchvision

print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("opencv      :", cv2.__version__)
print("numpy       :", np.__version__)
print("--- cuda check ---")
print("torch.version.cuda        :", torch.version.cuda)
print("cudnn.version()           :", torch.backends.cudnn.version())
print("cuda.is_available()       :", torch.cuda.is_available())
print("cudnn.is_available()      :", torch.backends.cudnn.is_available())
print("cuda.device_count()       :", torch.cuda.device_count())
print("device name               :", torch.cuda.get_device_name(0))
print("capability                :", torch.cuda.get_device_capability(0))