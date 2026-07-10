# 第 3 课：torchreid 项目结构

## 1. 本课目标

学完本课，你需要能够回答下面几个问题：

```text
1. deep-person-reid 项目的核心目录有哪些？
2. 训练入口在哪里？
3. 配置文件在哪里？
4. 数据集、模型、训练引擎、评价指标分别在哪些目录？
5. 如果想读懂整个训练流程，应该按什么顺序读源码？
```

本课的重点是建立“项目地图”。当你知道每个目录负责什么之后，后面读源码会轻松很多。

## 2. 为什么要先看项目结构

很多人学习深度学习项目时，一上来就打开模型文件，比如：

```text
torchreid/models/resnet.py
torchreid/models/osnet.py
```

这样很容易迷路。

更好的方式是先问：

```text
程序从哪里开始？
配置从哪里来？
数据在哪里读？
模型在哪里建？
loss 在哪里算？
测试指标在哪里算？
结果在哪里保存？
```

也就是说，先看整体流程，再看局部实现。

## 3. 项目根目录概览

你的项目路径是：

```text
D:\code\deep-person-reid-master
```

核心结构大致是：

```text
deep-person-reid-master
├── configs
├── docs
├── projects
├── scripts
├── tools
├── torchreid
├── tutorial
├── README.rst
├── requirements.txt
└── setup.py
```

其中最重要的是：

```text
configs
scripts
torchreid
tutorial
README.rst
setup.py
```

## 4. README.rst：项目说明书

路径：

```text
README.rst
```

作用：

```text
介绍项目功能
说明安装方式
给出训练和测试示例
解释常见使用方法
```

你之前复现 Market1501 baseline 时，README 中最关键的信息是：

```text
训练入口：scripts/main.py
配置目录：configs/
安装方式：python setup.py develop
```

虽然我们在 Windows + 新版 setuptools 下用了 `.pth` 等价方式注册项目路径，但 README 仍然是理解项目推荐用法的第一站。

## 5. configs：实验配置目录

路径：

```text
configs
```

作用：

```text
存放不同模型和训练方案的 YAML 配置文件
```

例如：

```text
configs/im_r50_softmax_256x128_amsgrad.yaml
configs/im_r50fc512_softmax_256x128_amsgrad.yaml
configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml
configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml
```

你之前使用的是：

```text
configs/im_r50_softmax_256x128_amsgrad.yaml
```

这个配置指定了：

```yaml
model:
  name: 'resnet50_fc512'

data:
  sources: ['market1501']
  targets: ['market1501']
  height: 256
  width: 128

loss:
  name: 'softmax'

train:
  optim: 'amsgrad'
  lr: 0.0003
  max_epoch: 60
```

以后你做实验，优先改配置或命令行参数，不要急着改源码。

## 6. scripts：训练和测试入口

路径：

```text
scripts
```

里面最重要的两个文件：

```text
scripts/main.py
scripts/default_config.py
```

### 6.1 scripts/main.py

这是项目统一训练和测试入口。

你之前所有训练、测试、visrank 命令最终都运行的是：

```text
scripts/main.py
```

它主要做这些事：

```text
1. 解析命令行参数
2. 读取默认配置
3. 合并 YAML 配置
4. 合并命令行覆盖参数
5. 构建 DataManager
6. 构建模型
7. 构建优化器和学习率调度器
8. 构建 Engine
9. 启动训练或测试
```

可以把它理解成总指挥。

### 6.2 scripts/default_config.py

这个文件定义默认配置。

例如：

```python
cfg.data.root = 'reid-data'
cfg.data.sources = ['market1501']
cfg.data.targets = ['market1501']
cfg.data.height = 256
cfg.data.width = 128
cfg.train.max_epoch = 60
cfg.test.evaluate = False
```

你在命令行里写：

```text
test.evaluate True
test.batch_size 64
data.save_dir log\test_loaded_weight
```

其实就是覆盖这里的默认配置。

## 7. torchreid：项目核心代码

路径：

```text
torchreid
```

这是整个项目的核心 Python 包。

主要结构：

```text
torchreid
├── data
├── engine
├── losses
├── metrics
├── models
├── optim
└── utils
```

下面逐个看。

## 8. torchreid/data：数据模块

路径：

```text
torchreid/data
```

作用：

```text
负责数据集读取、数据增强、DataLoader 构建
```

重要文件：

```text
torchreid/data/datamanager.py
torchreid/data/transforms.py
torchreid/data/datasets/__init__.py
torchreid/data/datasets/image/market1501.py
```

### 8.1 datamanager.py

DataManager 是数据总管。

它负责：

```text
读取 source 数据集
读取 target 数据集
构建 train loader
构建 query loader
构建 gallery loader
打印数据集统计信息
```

你看到的日志：

```text
=> Loaded Market1501
subset   | # ids | # images | # cameras
train    |   751 |    12936 |         6
query    |   750 |     3368 |         6
gallery  |   751 |    15913 |         6
```

就是 DataManager 和 Dataset 类一起完成的。

### 8.2 transforms.py

负责图像预处理和数据增强。

常见操作：

```text
resize
random flip
random crop
random erase
to tensor
normalize
```

你之前关心的输入尺寸：

```text
data.height
data.width
```

最终会影响这里的 resize。

### 8.3 datasets/image/market1501.py

负责读取 Market1501。

它会解析：

```text
bounding_box_train
query
bounding_box_test
```

并从文件名中解析：

```text
pid
camid
```

## 9. torchreid/models：模型模块

路径：

```text
torchreid/models
```

作用：

```text
定义各种 ReID 模型
```

常见文件：

```text
torchreid/models/resnet.py
torchreid/models/osnet.py
torchreid/models/__init__.py
```

你之前使用的模型：

```text
resnet50_fc512
```

主要在：

```text
torchreid/models/resnet.py
```

OSNet 在：

```text
torchreid/models/osnet.py
```

模型构建入口在：

```python
torchreid.models.build_model(...)
```

在 `scripts/main.py` 中对应代码是：

```python
model = torchreid.models.build_model(
    name=cfg.model.name,
    num_classes=datamanager.num_train_pids,
    loss=cfg.loss.name,
    pretrained=cfg.model.pretrained,
    use_gpu=cfg.use_gpu
)
```

## 10. torchreid/losses：损失函数模块

路径：

```text
torchreid/losses
```

作用：

```text
定义训练时用的 loss
```

常见文件：

```text
torchreid/losses/cross_entropy_loss.py
torchreid/losses/hard_mine_triplet_loss.py
```

对应两类常见 ReID 训练方式：

```text
softmax classification
triplet metric learning
```

你当前 baseline 用的是：

```yaml
loss:
  name: 'softmax'
```

所以主要用交叉熵分类损失。

## 11. torchreid/optim：优化器和学习率

路径：

```text
torchreid/optim
```

作用：

```text
构建 optimizer 和 lr scheduler
```

重要文件：

```text
torchreid/optim/optimizer.py
torchreid/optim/lr_scheduler.py
```

你配置里的：

```yaml
train:
  optim: 'amsgrad'
  lr: 0.0003
  lr_scheduler: 'single_step'
  stepsize: [20]
```

会在这里转成真正的 PyTorch optimizer 和 scheduler。

## 12. torchreid/engine：训练和测试引擎

路径：

```text
torchreid/engine
```

作用：

```text
负责训练循环、测试流程、特征提取、保存模型
```

重要文件：

```text
torchreid/engine/engine.py
torchreid/engine/image/softmax.py
torchreid/engine/image/triplet.py
```

### 12.1 engine.py

通用训练和测试逻辑。

负责：

```text
run()
test()
feature extraction
distance matrix
metrics
visrank
checkpoint save
```

你看到的测试日志：

```text
Extracting features from query set ...
Extracting features from gallery set ...
Computing distance matrix ...
Computing CMC and mAP ...
```

主要来自这里。

### 12.2 image/softmax.py

图像 ReID + softmax loss 的训练逻辑。

你当前 baseline 用的是：

```text
ImageSoftmaxEngine
```

它负责每个 batch 的：

```text
前向传播
计算 loss
反向传播
更新参数
统计训练 acc
```

## 13. torchreid/metrics：评价指标模块

路径：

```text
torchreid/metrics
```

作用：

```text
计算距离、Rank-k、mAP
```

重要文件：

```text
torchreid/metrics/distance.py
torchreid/metrics/rank.py
```

### 13.1 distance.py

负责计算 query feature 和 gallery feature 之间的距离。

常见距离：

```text
euclidean
cosine
```

### 13.2 rank.py

负责计算：

```text
CMC
Rank-1
Rank-5
Rank-10
mAP
```

你看到的结果：

```text
mAP: 22.6%
Rank-1  : 40.6%
Rank-5  : 62.9%
Rank-10 : 72.4%
Rank-20 : 80.6%
```

就是这里计算出来的。

## 14. torchreid/utils：工具函数

路径：

```text
torchreid/utils
```

作用：

```text
日志、checkpoint、可视化、工具函数
```

重要文件：

```text
torchreid/utils/tools.py
torchreid/utils/torchtools.py
torchreid/utils/reidtools.py
```

### 14.1 tools.py

常见工具函数。

例如：

```text
创建目录
下载文件
收集环境信息
```

### 14.2 torchtools.py

和 PyTorch 权重相关的工具。

例如：

```text
保存 checkpoint
加载 checkpoint
加载 pretrained weights
```

你之前遇到 PyTorch 2.6+ `weights_only` 的兼容问题，就是在这里处理的。

### 14.3 reidtools.py

ReID 可视化工具。

`visrank` 结果就是这里生成的。

输出目录类似：

```text
log\xxx\visrank_market1501
```

## 15. tools：额外工具脚本

路径：

```text
tools
```

作用：

```text
放一些模型导出、结果解析、辅助工具
```

例如 README 中提到的模型导出脚本：

```text
tools/export.py
```

初学阶段不需要优先看这个目录。

## 16. projects：研究项目示例

路径：

```text
projects
```

作用：

```text
存放基于 torchreid 扩展出来的研究项目
```

初学阶段可以先跳过。

等你理解主框架后，再看这些项目会更有意义。

## 17. tutorial：你的学习笔记目录

路径：

```text
tutorial
```

这是我们新建的学习目录。

当前已有：

```text
tutorial/reid_30_lessons_outline.md
tutorial/lesson_01_what_is_reid.md
tutorial/lesson_02_market1501_structure.md
tutorial/lesson_03_torchreid_project_structure.md
```

后续每一课都会放在这里。

## 18. 一次训练命令对应的源码流向

以你之前的训练命令为例：

```powershell
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py --config-file configs\im_r50_softmax_256x128_amsgrad.yaml --root D:\code\datasets data.workers 0 train.max_epoch 1 data.save_dir log\demo
```

大致执行流程是：

```text
scripts/main.py
-> get_default_config()
-> cfg.merge_from_file(config yaml)
-> cfg.merge_from_list(command opts)
-> ImageDataManager
-> Market1501 dataset
-> build_model()
-> build_optimizer()
-> build_lr_scheduler()
-> ImageSoftmaxEngine
-> engine.run()
-> train
-> test
-> save checkpoint
```

这条链就是你读源码时最重要的主线。

## 19. 推荐源码阅读顺序

不要一上来就读全部文件。

建议顺序：

```text
1. scripts/main.py
2. scripts/default_config.py
3. configs/im_r50_softmax_256x128_amsgrad.yaml
4. torchreid/data/datasets/image/market1501.py
5. torchreid/data/datamanager.py
6. torchreid/models/resnet.py
7. torchreid/engine/image/softmax.py
8. torchreid/engine/engine.py
9. torchreid/metrics/distance.py
10. torchreid/metrics/rank.py
11. torchreid/utils/reidtools.py
```

这个顺序是从主流程到细节，比较不容易迷路。

## 20. 本课动手任务

### 任务 1：查看项目根目录

在 PowerShell 中执行：

```powershell
Get-ChildItem D:\code\deep-person-reid-master
```

观察：

```text
configs
scripts
torchreid
tutorial
README.rst
setup.py
```

### 任务 2：查看 scripts 目录

```powershell
Get-ChildItem D:\code\deep-person-reid-master\scripts
```

确认：

```text
main.py
default_config.py
```

### 任务 3：查看 torchreid 核心目录

```powershell
Get-ChildItem D:\code\deep-person-reid-master\torchreid
```

观察：

```text
data
engine
losses
metrics
models
optim
utils
```

### 任务 4：打开 main.py 找主流程

重点找这些代码：

```python
datamanager = build_datamanager(cfg)
model = torchreid.models.build_model(...)
optimizer = torchreid.optim.build_optimizer(...)
scheduler = torchreid.optim.build_lr_scheduler(...)
engine = build_engine(...)
engine.run(...)
```

如果你能在 `main.py` 中找到这些语句，就已经抓住了项目主干。

## 21. 和老虎 ReID 的关系

以后你迁移到老虎数据集时，最可能改动的是：

```text
torchreid/data/datasets/image/
torchreid/data/datasets/image/__init__.py
torchreid/data/datasets/__init__.py
configs/
```

最开始不建议改：

```text
models
engine
metrics
```

因为你第一阶段的目标应该是：

```text
用已有模型和训练流程，先把自定义数据集读进去并跑通 baseline。
```

等 baseline 稳定后，再考虑改模型或 loss。

## 22. 本课小结

本课你需要记住：

```text
1. scripts/main.py 是训练和测试总入口。
2. scripts/default_config.py 定义默认配置。
3. configs 存放实验配置。
4. torchreid/data 负责数据读取和 DataLoader。
5. torchreid/models 负责模型。
6. torchreid/losses 负责 loss。
7. torchreid/optim 负责优化器和学习率。
8. torchreid/engine 负责训练和测试流程。
9. torchreid/metrics 负责 mAP 和 Rank-k。
10. torchreid/utils 负责日志、权重、可视化等工具。
```

下一课建议学习：

```text
第 4 课：环境配置与依赖理解
```

