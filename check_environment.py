import torch
import transformers
import sklearn
import pandas
import numpy
import matplotlib

print("Environment check successful!")
print("-----------------------------")

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)

print("-----------------------------")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
