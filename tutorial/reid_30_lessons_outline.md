# ReID 重识别 30 课学习大纲

本课程以 `deep-person-reid / torchreid` 项目为主线，目标是从“能跑通代码”逐步过渡到“能理解 ReID 原理、能读懂源码、能改实验、能迁移到自己的数据集”。

建议每节课按照以下节奏学习：

```text
1. 先理解概念
2. 再阅读项目关键文件
3. 最后运行或修改一个小实验
```

课程主线：

```text
数据集 -> DataManager -> Model -> Loss -> Engine -> Feature -> Distance -> Metrics -> Visrank -> 自定义数据集
```

## 第 1 课：什么是重识别 ReID

### 学习目标

- 理解 ReID 和分类、检测、跟踪的区别。
- 理解 query、gallery、train 的含义。
- 建立“ReID 本质是图像检索任务”的直觉。

### 课程介绍

ReID 的任务是给定一张 query 图片，在 gallery 图片库中找出同一身份的图片。训练时模型通常像分类模型一样学习身份类别，测试时则输出特征向量，通过距离排序完成检索。

### 项目关键文件

- `README.rst`
- `scripts/main.py`
- `torchreid/engine/engine.py`

## 第 2 课：Market1501 数据集结构

### 学习目标

- 理解 Market1501 的目录结构。
- 理解 `bounding_box_train`、`query`、`bounding_box_test` 的作用。
- 理解图片文件名中的 `pid` 和 `camid`。

### 课程介绍

Market1501 是 Person ReID 中最常用的数据集之一。它将图片划分为训练集、查询集和图库集。模型训练时使用训练集，测试时用 query 去 gallery 中检索同一身份。

### 项目关键文件

- `torchreid/data/datasets/image/market1501.py`
- `torchreid/data/datasets/dataset.py`
- `D:\code\datasets\market1501\Market-1501-v15.09.15`

## 第 3 课：torchreid 项目结构

### 学习目标

- 熟悉 deep-person-reid 的整体目录。
- 知道训练入口、配置文件、模型、数据、评估代码分别在哪里。
- 建立源码阅读路线。

### 课程介绍

torchreid 将项目拆成数据、模型、损失、优化器、训练引擎、工具函数等模块。学习时不要从模型文件硬啃，而应从 `scripts/main.py` 的主流程开始。

### 项目关键文件

- `scripts/main.py`
- `scripts/default_config.py`
- `configs/`
- `torchreid/data/`
- `torchreid/models/`
- `torchreid/engine/`
- `torchreid/metrics/`

## 第 4 课：环境配置与依赖理解

### 学习目标

- 理解 conda 环境、PyTorch、torchvision、CUDA 的关系。
- 掌握验证 PyTorch 和 CUDA 的命令。
- 理解 Windows 下常见兼容问题。

### 课程介绍

复现深度学习项目时，环境稳定比盲目升级更重要。Windows 下尤其要注意路径、编码、DataLoader 多进程、旧项目与新 PyTorch 的兼容性。

### 项目关键文件

- `requirements.txt`
- `setup.py`
- `scripts/main.py`
- `torchreid/utils/tools.py`
- `torchreid/utils/torchtools.py`

## 第 5 课：跑通第一个 baseline

### 学习目标

- 使用 ResNet50 在 Market1501 上训练 1 epoch。
- 理解 smoke test 的意义。
- 找到日志、权重和测试结果。

### 课程介绍

第一阶段不追求高精度，而是确认数据读取、模型构建、loss、反向传播、checkpoint 保存和测试流程全部正常。

### 项目关键文件

- `configs/im_r50_softmax_256x128_amsgrad.yaml`
- `scripts/main.py`
- `log/resnet50_market1501_smoke/`

## 第 6 课：从 scripts/main.py 看主流程

### 学习目标

- 理解 `main.py` 的执行顺序。
- 理解 `build_datamanager`、`build_model`、`build_engine` 的关系。
- 知道训练真正从哪里开始。

### 课程介绍

`scripts/main.py` 是整个项目最重要的入口文件。它负责读取配置、构建数据、模型、优化器、学习率调度器和训练引擎。

### 项目关键文件

- `scripts/main.py`
- `scripts/default_config.py`
- `torchreid/engine/engine.py`

## 第 7 课：配置系统 yacs

### 学习目标

- 理解默认配置和 YAML 配置的关系。
- 掌握命令行覆盖配置的方法。
- 学会用配置管理实验。

### 课程介绍

torchreid 使用 yacs 管理配置。默认配置在 `default_config.py`，实验配置在 `configs/`，命令行参数可以继续覆盖 YAML 中的设置。

### 项目关键文件

- `scripts/default_config.py`
- `configs/im_r50_softmax_256x128_amsgrad.yaml`
- `configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml`

## 第 8 课：DataManager 是什么

### 学习目标

- 理解 Dataset 和 DataLoader 的区别。
- 理解 ImageDataManager 的职责。
- 理解 source dataset 和 target dataset。

### 课程介绍

DataManager 是数据进入模型前的总调度器。它负责创建训练集、测试集、transform、sampler 和 dataloader。

### 项目关键文件

- `torchreid/data/datamanager.py`
- `torchreid/data/datasets/__init__.py`
- `torchreid/data/datasets/image/market1501.py`

## 第 9 课：Market1501 数据类源码

### 学习目标

- 看懂 `market1501.py` 如何读取图片。
- 理解 `relabel=True` 的作用。
- 理解 junk pid 的处理。

### 课程介绍

Market1501 的图片名自带身份和摄像头信息，数据类通过正则表达式解析文件名，构造 `(img_path, pid, camid)` 形式的数据列表。

### 项目关键文件

- `torchreid/data/datasets/image/market1501.py`
- `torchreid/data/datasets/dataset.py`

## 第 10 课：图像预处理与数据增强

### 学习目标

- 理解 resize、random flip、random erase、normalization。
- 区分训练 transform 和测试 transform。
- 知道数据增强对 ReID 的意义。

### 课程介绍

训练阶段通常使用随机增强提升泛化能力，测试阶段则保持确定性处理。ReID 中常见增强包括随机翻转、随机裁剪和随机擦除。

### 项目关键文件

- `torchreid/data/transforms.py`
- `scripts/default_config.py`
- `configs/*.yaml`

## 第 11 课：ReID baseline 的整体结构

### 学习目标

- 理解 backbone、pooling、embedding、classifier。
- 理解训练时分类、测试时检索。
- 理解 feature 在 ReID 中的核心作用。

### 课程介绍

ReID baseline 通常使用 CNN 提取特征，训练时通过分类器学习身份类别，测试时丢弃分类器，只使用特征向量进行距离检索。

### 项目关键文件

- `torchreid/models/`
- `torchreid/models/resnet.py`
- `torchreid/engine/image/softmax.py`

## 第 12 课：ResNet50 在 ReID 中怎么用

### 学习目标

- 理解 ImageNet 预训练的作用。
- 理解 `resnet50` 和 `resnet50_fc512` 的区别。
- 理解为什么 1 epoch 预训练模型效果明显更好。

### 课程介绍

ResNet50 是经典 CNN backbone。`resnet50_fc512` 在 backbone 后增加 512 维全连接特征层，更适合输出 ReID embedding。

### 项目关键文件

- `torchreid/models/resnet.py`
- `configs/im_r50_softmax_256x128_amsgrad.yaml`
- `log/resnet50_market1501_pretrained_1epoch/`

## 第 13 课：特征向量是什么

### 学习目标

- 理解图片如何变成向量。
- 理解类内距离和类间距离。
- 理解 embedding space 的直观含义。

### 课程介绍

模型输出的 feature 是 ReID 的核心。理想情况下，同一身份的图片特征距离近，不同身份的图片特征距离远。

### 项目关键文件

- `torchreid/engine/engine.py`
- `torchreid/metrics/distance.py`
- `torchreid/models/resnet.py`

## 第 14 课：分类头 classifier 的作用

### 学习目标

- 理解训练集身份数和 classifier 输出维度的关系。
- 理解测试时为什么不用 classifier。
- 理解迁移到新数据集时为什么要重新构建分类头。

### 课程介绍

在 Market1501 中训练身份数是 751，所以分类器输出 751 类。测试时 query/gallery 身份不完全等同于训练分类任务，因此使用 embedding 做检索。

### 项目关键文件

- `torchreid/models/resnet.py`
- `scripts/main.py`
- `torchreid/engine/image/softmax.py`

## 第 15 课：OSNet 模型入门

### 学习目标

- 理解 OSNet 的基本思想。
- 理解 omni-scale feature learning。
- 知道 OSNet 和 ResNet baseline 的区别。

### 课程介绍

OSNet 是 torchreid 项目的核心模型，重点是同时学习多尺度特征，更适合处理 ReID 中姿态、尺度和局部细节变化。

### 项目关键文件

- `torchreid/models/osnet.py`
- `configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml`
- `README.rst`

## 第 16 课：Softmax Loss baseline

### 学习目标

- 理解把 ReID 当分类任务训练的方式。
- 理解 cross entropy 和 label smoothing。
- 理解 softmax baseline 的优点和局限。

### 课程介绍

Softmax loss 将每个训练身份看作一个类别。它简单、稳定，是 ReID baseline 的常用起点。

### 项目关键文件

- `torchreid/engine/image/softmax.py`
- `torchreid/losses/cross_entropy_loss.py`
- `scripts/default_config.py`

## 第 17 课：Triplet Loss 入门

### 学习目标

- 理解 anchor、positive、negative。
- 理解 margin 的作用。
- 理解 hard mining 的意义。

### 课程介绍

Triplet Loss 更贴近检索目标，直接约束同类样本距离更近、异类样本距离更远，是 ReID 中非常重要的损失函数。

### 项目关键文件

- `torchreid/engine/image/triplet.py`
- `torchreid/losses/hard_mine_triplet_loss.py`
- `scripts/default_config.py`

## 第 18 课：Softmax 和 Triplet 的区别

### 学习目标

- 比较分类学习和距离学习。
- 理解两种 loss 对特征空间的影响。
- 知道什么时候使用 softmax、triplet 或二者结合。

### 课程介绍

Softmax 更关注身份分类边界，Triplet 更关注样本间相对距离。实际 ReID 系统中经常结合两类目标。

### 项目关键文件

- `torchreid/engine/image/softmax.py`
- `torchreid/engine/image/triplet.py`
- `torchreid/losses/`

## 第 19 课：优化器与学习率

### 学习目标

- 理解 Adam、AMSGrad、SGD。
- 理解 learning rate、weight decay。
- 理解 scheduler 和 cosine schedule。

### 课程介绍

训练效果不只由模型决定，也强烈依赖优化器和学习率策略。学习率过大可能不收敛，过小则训练缓慢。

### 项目关键文件

- `torchreid/optim/optimizer.py`
- `torchreid/optim/lr_scheduler.py`
- `scripts/default_config.py`
- `configs/*.yaml`

## 第 20 课：训练日志怎么看

### 学习目标

- 理解 loss、acc、lr、time、data time、eta。
- 判断训练是否正常。
- 学会从日志中发现问题。

### 课程介绍

训练日志是模型状态的第一手证据。loss 是否下降、acc 是否上升、data time 是否异常，都能帮助判断问题位置。

### 项目关键文件

- `torchreid/engine/engine.py`
- `torchreid/engine/image/softmax.py`
- `log/*/train.log-*`

## 第 21 课：测试阶段发生了什么

### 学习目标

- 理解 query 和 gallery 的特征提取。
- 理解距离矩阵。
- 理解测试阶段和训练阶段的区别。

### 课程介绍

测试时模型不再更新参数，而是对 query 和 gallery 提取特征，计算距离矩阵，再根据距离排序得到检索结果。

### 项目关键文件

- `torchreid/engine/engine.py`
- `torchreid/metrics/distance.py`
- `torchreid/metrics/rank.py`

## 第 22 课：距离度量

### 学习目标

- 理解 euclidean distance。
- 理解 cosine distance。
- 理解 normalize feature。
- 理解距离矩阵的形状和意义。

### 课程介绍

ReID 的检索过程依赖特征距离。每个 query 和每个 gallery 都会计算一个距离，最终形成 `num_query x num_gallery` 的矩阵。

### 项目关键文件

- `torchreid/metrics/distance.py`
- `torchreid/engine/engine.py`
- `scripts/default_config.py`

## 第 23 课：CMC 与 Rank-k

### 学习目标

- 理解 Rank-1、Rank-5、Rank-10。
- 理解 CMC 曲线。
- 理解 Rank-k 的优点和局限。

### 课程介绍

Rank-k 表示正确目标是否出现在前 k 个检索结果中。Rank-1 很直观，但不能完整反映所有正确匹配的排序质量。

### 项目关键文件

- `torchreid/metrics/rank.py`
- `torchreid/engine/engine.py`
- `log/*/test.log-*`

## 第 24 课：mAP 指标

### 学习目标

- 理解 Average Precision。
- 理解 mean Average Precision。
- 理解 mAP 和 Rank-1 的区别。

### 课程介绍

mAP 更关注所有正确匹配在排序列表中的整体位置，是 ReID 中非常重要的综合评价指标。

### 项目关键文件

- `torchreid/metrics/rank.py`
- `torchreid/engine/engine.py`

## 第 25 课：visrank 结果展示

### 学习目标

- 学会使用 `test.visrank True`。
- 理解 query 和 top-k gallery 的展示方式。
- 通过可视化分析模型错误。

### 课程介绍

visrank 是理解模型表现最直观的工具。它会为每个 query 保存 top-k 检索图，绿色边框表示匹配正确，红色边框表示匹配错误。

### 项目关键文件

- `torchreid/utils/reidtools.py`
- `torchreid/engine/engine.py`
- `log/resnet50_market1501_pretrained_1epoch_visrank/visrank_market1501/`

## 第 26 课：如何设计一个 baseline 实验

### 学习目标

- 理解 baseline 的意义。
- 学会一次只改一个变量。
- 学会保存命令、日志、权重和结果。

### 课程介绍

好的实验设计比盲目改模型更重要。baseline 是后续所有改进的参照物，必须保证可复现。

### 项目关键文件

- `configs/*.yaml`
- `scripts/main.py`
- `log/`

## 第 27 课：从 Market1501 迁移到自己的数据集

### 学习目标

- 理解自定义 ReID 数据集需要哪些信息。
- 学会设计 train/query/gallery。
- 理解 pid 和 camid 如何定义。

### 课程介绍

迁移到自己的数据集时，最重要的是保证数据格式清晰、身份编号稳定、训练集和测试集划分合理。

### 项目关键文件

- `torchreid/data/datasets/image/market1501.py`
- `torchreid/data/datasets/image/*.py`
- `torchreid/data/datasets/__init__.py`

## 第 28 课：自定义 Dataset 类

### 学习目标

- 参考 Market1501 写自己的 Dataset 类。
- 将自定义数据集注册到 torchreid。
- 先用小数据集跑通读取流程。

### 课程介绍

自定义 Dataset 类的核心是构造 train、query、gallery 三个列表，每个元素包含图片路径、身份 id 和摄像头 id。

### 项目关键文件

- `torchreid/data/datasets/image/market1501.py`
- `torchreid/data/datasets/image/__init__.py`
- `torchreid/data/datasets/__init__.py`

## 第 29 课：ReID 失败案例分析

### 学习目标

- 学会从 visrank 中分析失败原因。
- 识别姿态、遮挡、光照、背景、视角造成的问题。
- 将错误分析转化为下一步实验。

### 课程介绍

ReID 的难点往往不在代码，而在数据变化。失败案例能告诉你模型到底没有学到什么。

### 项目关键文件

- `log/*/visrank_market1501/`
- `torchreid/utils/reidtools.py`
- `torchreid/metrics/rank.py`

## 第 30 课：完整项目实战总结

### 学习目标

- 串起完整 ReID 流程。
- 能独立复现 baseline。
- 能解释训练、测试、可视化结果。
- 为迁移到自定义数据集做准备。

### 课程介绍

最后一课将前面所有知识串起来：从数据准备到训练，从测试指标到可视化，从 baseline 到自定义数据集，形成完整的 ReID 实践闭环。

### 项目关键文件

- `scripts/main.py`
- `configs/`
- `torchreid/data/`
- `torchreid/models/`
- `torchreid/engine/`
- `torchreid/metrics/`
- `torchreid/utils/reidtools.py`
- `log/`

## 附录：推荐实践顺序

```text
1. 跑通 ResNet50 + Market1501 + 1 epoch
2. 打开 ImageNet 预训练再跑 1 epoch
3. 使用 test.evaluate True 独立测试
4. 使用 test.visrank True 生成可视化结果
5. 将 max_epoch 提高到 5 或 10
6. 换 OSNet 配置
7. 阅读 Market1501 数据集类
8. 准备自己的 ReID 数据集格式
9. 写自定义 Dataset 类
10. 训练自己的 baseline
```

## 附录：常用命令模板

### 训练 1 epoch

```bat
cd /d D:\code\deep-person-reid-master

F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py ^
  --config-file configs\im_r50_softmax_256x128_amsgrad.yaml ^
  --root D:\code\datasets ^
  data.workers 0 ^
  train.max_epoch 1 ^
  train.fixbase_epoch 0 ^
  train.batch_size 16 ^
  test.batch_size 64 ^
  data.save_dir log\resnet50_market1501_demo
```

### 独立测试

```bat
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py ^
  --config-file configs\im_r50_softmax_256x128_amsgrad.yaml ^
  --root D:\code\datasets ^
  data.workers 0 ^
  model.load_weights log\resnet50_market1501_demo\model\model.pth.tar-1 ^
  test.evaluate True ^
  data.save_dir log\resnet50_market1501_demo_test
```

### 生成 visrank 可视化

```bat
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py ^
  --config-file configs\im_r50_softmax_256x128_amsgrad.yaml ^
  --root D:\code\datasets ^
  data.workers 0 ^
  model.load_weights log\resnet50_market1501_demo\model\model.pth.tar-1 ^
  test.evaluate True ^
  test.visrank True ^
  test.visrank_topk 10 ^
  data.save_dir log\resnet50_market1501_demo_visrank
```
