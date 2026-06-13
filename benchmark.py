import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort
import numpy as np
import time

print(f"Loading models...")

# PyTorch
model_pytorch = models.resnet18(weights = None)
number_features_pytorch = model_pytorch.fc.in_features
model_pytorch.fc = nn.Linear(number_features_pytorch, 7)
model_pytorch.load_state_dict(torch.load("resnet18.pth", weights_only=True))
model_pytorch.eval()

# ONNX
ort_session = ort.InferenceSession("resnet18.onnx", providers=['CPUExecutionProvider'])

dummy_input_pytorch = torch.randn(1,3,224,224)
dummy_input_onnx = dummy_input_pytorch.numpy()

with torch.no_grad():
    output_pytorch = model_pytorch(dummy_input_pytorch).numpy()

output_onnx = ort_session.run(["output"], {"input": dummy_input_onnx})[0]

diff = np.abs(output_pytorch - output_onnx).mean()

print(f"Mean difference between PyTorch and ONNX: {diff} ")

iterations = 100
warmup = 20

for _ in range(warmup):
    with torch.no_grad():
        _ = model_pytorch(dummy_input_pytorch)
    _ = ort_session.run(["output"], {"input": dummy_input_onnx})

start_time = time.perf_counter()
for _ in range(iterations):
    with torch.no_grad():
        _ = model_pytorch(dummy_input_pytorch)
end_time = time.perf_counter()

pytorch_time = (end_time - start_time) / iterations

start_time = time.perf_counter()
for _ in range(iterations):
    _ = ort_session.run(["output"], {"input": dummy_input_onnx})
end_time = time.perf_counter()

onnx_time = (end_time - start_time) / iterations

print(f"PyTorch time: {pytorch_time} ms")
print(f"ONNX time: {onnx_time} ms")

if onnx_time > pytorch_time:
    print(f"PyTorch is faster than ONNX by {onnx_time - pytorch_time} ms")
else:
    print(f"ONNX is faster than PyTorch by {pytorch_time - onnx_time} ms")