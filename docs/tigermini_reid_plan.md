# TigerMini 自定义数据集重识别计划

本文档面向当前项目 `D:\code\deep-person-reid-master`，目标是把已经放在 `reid-data\TigerMini` 下的老虎数据集接入 `torchreid`，完成训练、评估和后续检索使用。

## 1. 当前数据现状

当前目录：

```text
reid-data/
└── TigerMini/
    ├── train/
    ├── query/
    └── gallery/
```

已检查到的数据规模：

```text
train   : 302 个身份，3557 张图片
query   : 201 个身份，402 张图片
gallery : 201 个身份，1997 张图片
```

文件名示例：

```text
train/video_001_0001.jpg
query/video_002_0001.jpg
gallery/video_002_0010.jpg
```

当前采用的解释规则：

```text
video_001_0001.jpg
      ^^^
      tiger id / pid

最后的 0001、0010、0096 等数字作为帧号或采样序号。
```

也就是说，当前认为 `video_XXX` 中的 `XXX` 就是老虎个体身份编号 `pid`。

## 2. ReID 任务定义

TigerMini 的重识别任务可以定义为：

```text
给定一张 query 老虎图片，在 gallery 图库中找出同一只老虎的图片。
```

训练阶段：

```text
train 目录中的每个 tiger id 作为一个分类类别。
模型学习把同一只老虎的特征拉近，把不同老虎的特征拉远。
```

测试阶段：

```text
query 中每张图片作为查询图。
gallery 中所有图片作为候选图库。
模型输出距离排序，计算 Rank-1、Rank-5、mAP 等指标。
```

当前拆分是合理的：`train` 身份和 `query/gallery` 身份分离，符合常见 ReID 评估设置。

## 3. 必须解决的 camid 问题

`torchreid` 的每条样本最终都必须是：

```python
(img_path, pid, camid)
```

其中：

```text
img_path -> 图片路径
pid      -> 身份 ID，即老虎个体编号
camid    -> 摄像头、视频源、拍摄地点或人为定义的视角编号
```

当前 TigerMini 文件名没有类似 Market1501 的 `c1/c2` 摄像头字段。需要特别注意：`torchreid` 的 Market1501 评估逻辑会过滤掉“同 pid 且同 camid”的 gallery 图片。如果 query 和 gallery 被全部设成同一个 `camid`，同身份 gallery 可能被全部过滤，导致评估失败或指标无意义。

当前采用推荐的最小可用策略：

```text
train   : camid = 0
query   : camid = 0
gallery : camid = 1
```

这样可以保证 query 与 gallery 中同一只老虎不会因为同 camid 被过滤掉，能先跑通训练和评估。

更规范的长期策略：

```text
如果原始数据来自不同视频、机位、地点或日期，应把这些来源编码进文件名或标注文件，
再用真实来源解析 camid。
```

例如：

```text
tiger_002_c01_0001.jpg
tiger_002_c02_0010.jpg
```

或者维护一个 CSV：

```text
filename,pid,camid,split
video_002_0001.jpg,2,0,query
video_002_0010.jpg,2,1,gallery
```

## 4. 接入方案

推荐新增一个 TigerMini 数据集类，而不是直接改 Market1501。

原因：

```text
1. TigerMini 的目录名是 train/query/gallery，不是 bounding_box_train/query/bounding_box_test。
2. TigerMini 的文件名格式是 video_XXX_YYYY.jpg，不包含 Market1501 的 cN 字段。
3. 单独数据集类更清晰，后续修改 camid 或标注格式不会影响 Market1501。
```

需要新增或修改的文件：

```text
torchreid/data/datasets/image/tigermini.py
torchreid/data/datasets/image/__init__.py
torchreid/data/datasets/__init__.py
configs/im_r50_softmax_256x128_tigermini.yaml
```

数据集类的核心逻辑：

```text
1. dataset_dir = 'TigerMini'
2. train_dir   = reid-data/TigerMini/train
3. query_dir   = reid-data/TigerMini/query
4. gallery_dir = reid-data/TigerMini/gallery
5. 用正则解析 video_XXX_YYYY.jpg 中的 XXX 作为 pid
6. train relabel=True，让训练类别变成连续编号
7. query/gallery relabel=False，保留原始 pid 用于评估
8. 按 split 设置 camid：train/query=0，gallery=1
```

建议解析规则：

```python
pattern = re.compile(r'video_(\d+)_(\d+)')
pid = int(match.group(1))
```

## 5. 推荐训练配置

当前目标是先用预训练 ResNet 跑通 TigerMini 的完整流程。项目里的 ResNet 配置文件使用的是 `resnet50_fc512`，并且默认 `pretrained: True`。

配置文件建议从这里复制：

```text
configs/im_r50_softmax_256x128_amsgrad.yaml
```

新建：

```text
configs/im_r50_softmax_256x128_tigermini.yaml
```

推荐初始配置：

```yaml
model:
  name: 'resnet50_fc512'
  pretrained: True

data:
  type: 'image'
  sources: ['tigermini']
  targets: ['tigermini']
  height: 256
  width: 128
  combineall: False
  transforms: ['random_flip']
  save_dir: 'log/resnet50_tigermini_softmax'
  workers: 0

loss:
  name: 'softmax'
  softmax:
    label_smooth: True

train:
  optim: 'amsgrad'
  lr: 0.0003
  max_epoch: 60
  batch_size: 32
  fixbase_epoch: 5
  open_layers: ['classifier']
  lr_scheduler: 'single_step'
  stepsize: [20]
  print_freq: 10

test:
  batch_size: 100
  dist_metric: 'euclidean'
  normalize_feature: False
  evaluate: False
  eval_freq: -1
  rerank: False
  ranks: [1, 5, 10, 20]
```

说明：

```text
1. 先沿用项目 ResNet 配置的主要超参数，减少变量。
2. `pretrained=True` 会使用预训练 ResNet 权重，更适合 TigerMini 这种小数据集。
3. batch_size 先用 32，比较稳。
4. workers 在 Windows 上先设为 0，减少多进程 DataLoader 问题。
5. `eval_freq=-1` 表示训练结束后评估一次，先把完整流程跑通。
```

## 6. 训练命令

在项目根目录运行：

```powershell
cd D:\code\deep-person-reid-master

F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py `
  --config-file configs\im_r50_softmax_256x128_tigermini.yaml `
  --root reid-data
```

如果 GPU 显存不足：

```powershell
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py `
  --config-file configs\im_r50_softmax_256x128_tigermini.yaml `
  --root reid-data `
  train.batch_size 16 test.batch_size 32
```

训练成功时，日志中应看到类似：

```text
=> Loaded TigerMini
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   302 |     3557 |         1
  query    |   201 |      402 |         1
  gallery  |   201 |     1997 |         1
  ----------------------------------------
```

注意：如果按推荐策略 query=0、gallery=1，那么 query 和 gallery 分别显示 1 个 camera 是正常的；评估合并使用时 camid 仍然不同。

## 7. 单独评估命令

训练完成后，模型一般保存在：

```text
log/resnet50_tigermini_softmax/model/
```

假设最佳或最后 checkpoint 是：

```text
log/resnet50_tigermini_softmax/model/model.pth.tar-60
```

只评估命令：

```powershell
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py `
  --config-file configs\im_r50_softmax_256x128_tigermini.yaml `
  --root reid-data `
  model.load_weights log\resnet50_tigermini_softmax\model\model.pth.tar-60 `
  test.evaluate True `
  test.batch_size 100 `
  data.save_dir log\resnet50_tigermini_eval
```

重点观察：

```text
mAP
Rank-1
Rank-5
Rank-10
```

## 8. 验收标准

第一阶段只要求跑通：

```text
1. 数据集能被注册为 tigermini。
2. DataManager 能打印出正确的 train/query/gallery 数量。
3. 能完成至少 1 个 epoch 训练。
4. 能完成一次 query-gallery 评估。
5. 输出 mAP 和 CMC Rank 指标。
```

第二阶段再追求效果：

```text
1. Rank-1 明显高于随机检索。
2. 可视化 top-k 结果中，同一只老虎排在前列。
3. 错误样例集中分析：姿态、遮挡、裁剪质量、背景相似、帧间重复。
```

## 9. 效果提升方向

数据层面：

```text
1. 抽查每个 video_XXX 内部图片是否保持同一只老虎，及时清理混入样本。
2. 清理误检、空图、严重遮挡和过小 crop。
3. 尽量补充跨场景、跨日期、跨姿态样本。
4. 如果 query/gallery 来自同一段连续视频，评估可能偏容易；应补充更真实的跨来源 gallery。
```

模型层面：

```text
1. 先用预训练 resnet50_fc512 跑通完整流程。
2. 数据量较小时，保持 pretrained=True。
3. softmax 跑通后，再尝试 triplet 或 softmax+triplet。
4. 基线跑通后，可对比 normalize_feature=True/False。
5. 评估时可尝试 test.rerank True，但应同时记录未 rerank 的基线。
```

配置层面：

```text
1. 若过拟合明显，降低 max_epoch 或增加数据增强。
2. 若训练 loss 不下降，检查 pid 解析和 relabel。
3. 若评估报 all query identities do not appear in gallery，优先检查 query/gallery 的 pid 和 camid。
4. 若 DataLoader 卡住，Windows 上保持 data.workers 0。
```

## 10. 后续实际检索流程

训练评估只是第一步。真正使用 TigerMini 做重识别时，可以做成以下流程：

```text
1. 用训练好的模型提取 gallery 全部图片特征。
2. 保存 gallery_features、gallery_paths、gallery_pids。
3. 对新输入的 query 图片提取特征。
4. 计算 query 与 gallery 的距离。
5. 返回 top-k 最相似图片路径和距离。
6. 可选：把 top-k 结果保存为可视化图片。
```

建议后续新增脚本：

```text
tools/tigermini_search.py
```

输入：

```powershell
--weights 训练好的模型
--query 单张老虎图片
--gallery-dir reid-data\TigerMini\gallery
--topk 10
```

输出：

```text
top-10 图片路径、距离分数、可视化结果图。
```

## 11. 推荐实施顺序

```text
1. 新增 tigermini.py 数据集类。
2. 在 image/__init__.py 导出 TigerMini。
3. 在 datasets/__init__.py 注册 'tigermini'。
4. 基于 ResNet 配置新增 configs/im_r50_softmax_256x128_tigermini.yaml。
5. 先运行 1 个 epoch 验证数据读取和训练流程。
6. 跑完整 60 epoch ResNet 基线训练。
7. 只评估 best/last checkpoint。
8. 开启 visrank 或自写 top-k 检索脚本查看结果。
9. 根据错误样例调整数据清洗、camid 标注和训练配置。
```

## 12. 当前最重要的风险

```text
1. camid 缺失：会直接影响评估是否有效。
2. 当前已假定 video_XXX 是老虎个体 ID；如果后续发现它只是视频编号，需要重新标注真实个体 ID。
3. query/gallery 是否同源：如果来自同一段视频的相邻帧，指标可能虚高。
4. 小数据集过拟合：训练集 302 个身份、3557 张图，建议优先用预训练模型和较小学习率。
```
