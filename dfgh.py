import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("GPU is not available, using CPU.")


import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
