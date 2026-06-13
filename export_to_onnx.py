import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights = None)
num_fts = model.fc.in_features
model.fc = nn.Linear(num_fts,7)

model.load_state_dict(torch.load("resnet18.pth", weights_only=True))

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_file_path = "resnet18.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"},
                  "output": {0: "batch_size"}}
)

print(f'Model saved to {onnx_file_path}')