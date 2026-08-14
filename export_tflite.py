import os
import sys
from collections import OrderedDict
import torch

sys.path.append(os.getcwd())
from src.model_lib.MiniFASNet import MiniFASNetV2

# 选取 80x80 超轻量模型
model_path = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'

print(f"正在加载模型: {model_path}")
# 80x80 输入对应的 conv6_kernel 是 (5, 5)
model = MiniFASNetV2(conv6_kernel=(5, 5))

# 加载权重并处理多卡训练前缀
state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()

# 导出为 ONNX
dummy_input = torch.randn(1, 3, 80, 80)
onnx_filename = 'MiniFASNetV2.onnx'

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print(f"ONNX 导出成功: {onnx_filename}")
