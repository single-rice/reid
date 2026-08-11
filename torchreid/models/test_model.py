import torch
from torchreid.models import build_model

# 模拟输入输出查看shape
#model = build_model(name="resnet50", num_classes=751, loss="triplet")
model = build_model(name="resnet50", num_classes=751, loss="softmax")
model.train()

# 2. 构造模拟一批图片：batch=8, 3通道, 高256 宽128
dummy_input = torch.randn(8, 3, 256, 128)

# 3. 前向传播，打印输出shape
with torch.no_grad():
    output = model(dummy_input)

# 分情况打印
if isinstance(output, (list, tuple)):
    print("多输出：")
    for idx, o in enumerate(output):
        print(f"输出{idx} shape:", o.shape)
else:
    print("单输出 shape:", output.shape)