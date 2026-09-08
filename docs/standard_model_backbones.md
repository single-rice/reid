# 可修改的标准模型源码与预训练权重

本次撤回了上一版 `classification_backbones.py` 共享整网封装、六模型对比配置和实验 runner。
之前的视频采样、跨视频 Triplet 和带视频信息的 CSV 保持不变。已经下载的三个官方权重保留在
`pretrained/imagenet/`，可继续用于本次标准实现。

## 文件和模型入口

| 文件 | `build_model(name=...)` | 默认特征维度 | 本地可修改的主要模块 |
|---|---|---:|---|
| `torchreid/models/van.py` | `van_b0` / `van_b1` / `van_b2` / `van_b3` | 256 / 512 / 512 / 512 | LKA、Attention、Mlp、Block、OverlapPatchEmbed、各 stage |
| `torchreid/models/efficientnet.py` | `efficientnet_b0` | 1280 | MBConvConfig、MBConv、stage、pool、原始 dropout 分类头 |
| `torchreid/models/vit.py` | `vit_b_16` | 768 | MLPBlock、EncoderBlock、Encoder、patch 投影、CLS、位置编码 |
| `torchreid/models/swin_transformer.py` | `swin_t` / `swin_s` | 768 | PatchMerging、窗口注意力函数、ShiftedWindowAttention、SwinTransformerBlock |
| `torchreid/models/conformer.py` | `conformer_base_patch16` | 2112 | ConvBlock、Attention、Block、FCUDown/Up、ConvTransBlock、双分类头 |

Swin 的 Tiny/Small 按现有 `resnet.py` 中多个深度变体的方式放在同一个家族文件，构造入口分别独立。
VAN 主入口建议用 B2（Base），同时提供官方 B0–B3 以便改动深度和宽度。
各文件包含模型主体、模块定义、`model_urls`、`init_pretrained_weights` 和构造函数，
不是在内部调用 `torchvision.models.*()` 创建整网。

没有统一加 FC512、BNNeck 或新融合模块。保留官方原始特征维度，只将原分类头的类别数改为训练身份数。
底层仍复用 PyTorch 和 torchvision 的通用算子（例如 Conv2dNormActivation、MLP、StochasticDepth），
共享 `_standard_utils.py` 仅提供 DropPath、初始化检查与权重加载，不包含骨干网络。
VAN/Conformer 不需要额外安装 timm。

## 标准结构与输入

| 模型 | 标准配置 |
|---|---|
| VAN-B2 | embed dims `[64,128,320,512]`；depths `[3,3,12,3]`；MLP ratios `[8,8,4,4]` |
| EfficientNet-B0 | 标准 B0 的七组 MBConv、width/depth multiplier=1、1280 维池化、dropout=0.2 |
| ViT-B/16 | 224×224；patch=16；14×14 patch + 1 CLS；12 层、768 维、12 heads、MLP=3072 |
| Swin-T | 224×224；patch=4；window=7；depths `[2,2,6,2]`；heads `[3,6,12,24]` |
| Swin-S | 224×224；patch=4；window=7；depths `[2,2,18,2]`；heads `[3,6,12,24]` |
| Conformer-B/16 | 224×224；有效 patch=16；channel_ratio=6；Transformer 为 576 维、12 层、9 heads |

Conformer 的 Base 命名不等于 ViT-Base 参数：官方构造函数采用 576 维和 9 heads。
CNN 的最终特征为 1536 维，Transformer CLS 为 576 维；Re-ID 输出边界拼接成 2112 维，
不增加可训练投影。官方双分支、FCU 耦合及双分类头均保留。

ViT、Swin、Conformer 当前入口要求 224×224；输入检查会拒绝 128×256。
在数据配置中 resize，不在模型内部改变 patch 大小、做输入补丁或插值 ViT 位置编码。
Swin 原生的窗口处理、Conformer FCUUp 原生特征上采样属于原架构，均原样保留。
VAN 是大核卷积注意力网络，其 overlap patch embedding 也是原结构，不能改为 ViT 式分块。

例如使用现有配置并覆盖模型与图片大小：

```bash
python scripts/main.py --config-file configs/reid_r50fc512_softmax_128x256_wildtiger_amsgrad.yaml model.name swin_s data.height 224 data.width 224
```

也可在自己的 YAML 中设置：

```yaml
model:
  name: vit_b_16
  pretrained: True
data:
  height: 224
  width: 224
```

这是配置片段，其余数据、优化器、损失配置沿用项目现有配置。
若比较不同模型，应统一使用 224×224，并重新检查批大小和显存；不能沿用上一版 128×256 的显存测量。

## 预训练权重网址

下列地址来自 torchvision 或作者官方仓库，均为分类预训练权重。

| 模型 | 权重下载地址 |
|---|---|
| VAN-B0 | [van_tiny_754.pth.tar](https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar) |
| VAN-B1 | [van_small_811.pth.tar](https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar) |
| VAN-B2 | [van_base_828.pth.tar](https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar) |
| VAN-B3 | [van_large_839.pth.tar](https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar) |
| EfficientNet-B0 | [efficientnet_b0_rwightman-7f5810bc.pth](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth) |
| ViT-B/16 | [vit_b_16-c867db91.pth](https://download.pytorch.org/models/vit_b_16-c867db91.pth) |
| Swin-T | [swin_t-704ceda3.pth](https://download.pytorch.org/models/swin_t-704ceda3.pth) |
| Swin-S | [swin_s-5e29d889.pth](https://download.pytorch.org/models/swin_s-5e29d889.pth) |
| Conformer-B/16 | [作者 Google Drive 权重](https://drive.google.com/file/d/1oeQ9LSOGKEUaYGu7WTlUGl3KDsQIi0MA/view?usp=sharing) |

Conformer 作者还提供 [百度网盘](https://pan.baidu.com/s/1FL5XDAqHoimpUxNSunKq0w)，提取码 `b4z9`。
来源：[VAN 官方源码权重表](https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py)、
[EfficientNet 官方源码](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/efficientnet.html)、
[ViT 官方源码](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/vision_transformer.html)、
[Swin 官方源码](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/swin_transformer.html)、
[Conformer 官方发布页](https://github.com/pengzhiliang/Conformer#model-zoo)。

新构造函数的 `pretrained=True` 会加载上述指定权重，不使用可能随版本变化的 DEFAULT。
Google Drive 使用项目现有 gdown；若服务要求额外确认或下载不可用，可手动下载后传入本地路径。

```python
from torchreid.models import build_model

model = build_model(
    name='conformer_base_patch16',
    num_classes=323,
    loss='triplet',
    pretrained=True,
    pretrained_path='/path/to/Conformer_base_patch16.pth',
)
```

`pretrained_path` 是模型 Python 构造参数，不是此次恢复后的 `scripts/default_config.py` 配置项。
也可按已有 ConvNeXt 习惯，在 Python API 中直接传 `pretrained='/path/to/weights.pth'`。
权重键名和形状要求完整匹配骨干；只允许跳过类别数不同的原分类头，修改模块造成不兼容时会明确报错。
如果后续实验刻意替换某些模块，需要在对应模型的 `init_pretrained_weights` 中明确调整加载规则，
不会默认忽略任意缺失的骨干权重。ViT 兼容官方早期 MLP 键名，但不插值位置编码。

## 与现有训练代码的接口

普通模型训练时 `loss='softmax'` 返回 logits，`loss='triplet'` 返回 `(logits, features)`；
`eval()` 返回原始维度的 features。另提供 `forward_classification(x)`，即使在 eval 模式也返回分类结果，
便于与官方分类器做数值核对。

Conformer 保留两个 logits：

- softmax：`[cnn_logits, transformer_logits]`。
- triplet：`([cnn_logits, transformer_logits], concatenated_features)`。
- eval：`concatenated_features`。

项目已有 DeepSupervision 会分别计算两头的分类损失并取平均，不需要改损失引擎。
这保留了双头训练，但项目损失取平均的尺度不等于原论文训练代码的所有配方。
当前 `metrics.accuracy` 对列表只报告第一个头，所以日志 `acc` 是 CNN 分支准确率，
不是双分支融合分类准确率；检索使用双分支拼接特征。

若使用 `fixbase_epoch` 或 staged LR，需要填写实际顶层头名称：
EfficientNet 为 `classifier`；ViT 为 `heads`；VAN/Swin 为 `head`；
Conformer 为 `conv_cls_head` 和 `trans_cls_head`。不要给所有模型统一写不存在的 `classifier`。

## 验证与来源许可证

验证环境：PyTorch 2.8.0 / torchvision 0.23.0；没有修改用户的 Python 环境。

- EfficientNet-B0、ViT-B/16、Swin-T/S：与 torchvision 标准模型严格共享同一 state_dict，224×224 分类输出数值一致。
- VAN-B2、Conformer-B/16：与官方源文件构造的模型严格匹配 state_dict，同权重下分类输出最大误差为 0。
- 六种模型的 softmax、triplet、eval 接口及原生维度通过测试；矩形 Transformer 输入被拒绝。
- Conformer 双头接入现有跨视频训练引擎，反向传播覆盖两头和两个分支。
- 实际下载的 EfficientNet-B0、ViT-B/16、Swin-T 权重已在新实现加载并前向验证。
  VAN、Swin-S、Conformer 本次提供官方网址和兼容加载代码，未下载其权重文件；不能把结构核对等同于这些权重的实际加载验证。

运行：`python -m unittest discover -s tests`。
额外源码对齐记录：`log/standard_models_verification.json`。

EfficientNet/ViT/Swin 的架构代码来源为 torchvision v0.23.0，许可证见
`torchreid/models/licenses/torchvision.LICENSE`；VAN/Conformer 来源为作者官方仓库，
Apache-2.0 许可证分别保存在同目录的 `van.LICENSE`、`conformer.LICENSE`。
