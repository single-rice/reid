# 第 1 课：什么是重识别 ReID

## 1. 本课目标

学完本课，你需要能够回答下面几个问题：

```text
1. ReID 是什么任务？
2. ReID 和图像分类、目标检测、目标跟踪有什么区别？
3. query、gallery、train 分别是什么意思？
4. 为什么说 ReID 训练时像分类，测试时像检索？
5. torchreid 项目中，哪些文件负责训练、测试和结果展示？
```

本课不要求你马上理解所有模型细节。第一课的目标是先建立整体直觉：ReID 到底在解决什么问题。

## 2. ReID 的基本定义

ReID 是 Re-Identification 的缩写，中文通常叫“重识别”。

在 Person ReID 中，任务是：

```text
给定一张人的图片，在另一个图片库中找到同一个人的图片。
```

例如：

```text
query:
  一张摄像头 A 中出现的行人图片

gallery:
  摄像头 B、C、D 中拍到的大量行人图片

目标:
  在 gallery 中找出和 query 是同一个人的图片
```

所以 ReID 不是简单地判断“这张图是谁”，而是要解决“这张图和图库中的哪些图属于同一个身份”。

## 3. ReID 和其他视觉任务的区别

### 3.1 ReID 和图像分类的区别

图像分类任务通常是：

```text
输入一张图片 -> 输出一个固定类别
```

例如：

```text
猫 / 狗 / 老虎 / 汽车
```

ReID 训练时确实经常借用分类思路，把训练集中的每个身份当成一个类别。但是测试时，ReID 并不是只输出一个类别编号。

ReID 测试时更像这样：

```text
输入 query 图片
-> 提取 query 特征
-> 提取 gallery 所有图片特征
-> 计算 query 和每张 gallery 图片的距离
-> 按距离从近到远排序
-> 输出 top-k 检索结果
```

一句话总结：

```text
分类任务关心“这张图属于哪个类别”。
ReID 任务关心“图库中哪些图和这张图是同一个身份”。
```

### 3.2 ReID 和目标检测的区别

目标检测任务通常是：

```text
输入一张完整图片 -> 找出目标位置 -> 输出 bounding box 和类别
```

例如检测图片里有哪些人、车、动物。

而 ReID 通常默认输入已经是裁剪好的目标图：

```text
输入已经裁剪好的行人图 -> 提取身份特征 -> 做检索
```

在真实系统中，检测和 ReID 经常组合使用：

```text
原始视频帧
-> 目标检测器检测出人
-> 裁剪行人框
-> ReID 模型提取身份特征
-> 跨摄像头检索或关联
```

### 3.3 ReID 和目标跟踪的区别

目标跟踪通常是在同一个视频或同一个摄像头中连续跟踪目标：

```text
第 1 帧看到某个人
第 2 帧继续找到这个人
第 3 帧继续跟踪这个人
```

ReID 更强调跨摄像头、跨时间、跨场景的匹配：

```text
摄像头 A 的某个人
和摄像头 B 中某张图片
是否是同一个身份
```

在多目标跟踪中，ReID 特征常常被用来辅助身份关联。

## 4. ReID 中最重要的三个数据概念

ReID 数据集一般分为：

```text
train
query
gallery
```

### 4.1 train

`train` 是训练集。

模型在训练阶段会看到这些图片和它们对应的身份编号。

以 Market1501 为例：

```text
bounding_box_train
```

里面的图片用于训练模型。

训练时，模型学习：

```text
哪些图片属于同一个身份
哪些图片属于不同身份
如何把图片变成有区分度的特征向量
```

### 4.2 query

`query` 是查询集。

测试时，每一张 query 图片都会被拿出来，当作“我要找的人”。

以 Market1501 为例：

```text
query
```

里面的每一张图都会去 gallery 中检索。

### 4.3 gallery

`gallery` 是图库集。

测试时，模型会在 gallery 中为每个 query 找最相似的图片。

以 Market1501 为例：

```text
bounding_box_test
```

这个目录就是 gallery。

## 5. ReID 的完整流程

一个标准 ReID 流程可以写成：

```text
训练阶段:

train 图片
-> 数据增强
-> 模型提特征
-> classifier 分类
-> 计算 loss
-> 反向传播更新模型
```

测试阶段：

```text
query 图片
-> 模型提特征

gallery 图片
-> 模型提特征

query feature 和 gallery feature
-> 计算距离矩阵
-> 排序
-> 计算 Rank-1、Rank-5、mAP
-> 可视化 top-k 检索结果
```

这就是为什么常说：

```text
ReID 训练时像分类，测试时像检索。
```

## 6. 什么是特征向量

ReID 模型不会直接比较两张图片的像素。

它会先把图片变成一个向量。

例如你现在跑的 `resnet50_fc512` 模型，会输出类似这样的 512 维特征：

```text
image -> model -> feature vector
```

直观理解：

```text
同一个身份的图片，特征向量应该更接近。
不同身份的图片，特征向量应该更远。
```

测试阶段的检索就是基于这些特征距离完成的。

## 7. ReID 评价指标初步认识

本课只先认识两个最常见指标，后续课程会详细讲。

### 7.1 Rank-1

Rank-1 表示：

```text
对每张 query，gallery 排名第 1 的结果是否正确。
```

如果 Rank-1 是 40.6%，可以粗略理解为：

```text
大约 40.6% 的 query，第一名检索结果是正确身份。
```

### 7.2 mAP

mAP 更关注整体排序质量。

它不只看第一名，还会看所有正确匹配在整个排序列表中的位置。

一般来说：

```text
Rank-1 看第一眼是否找对。
mAP 看整体检索排序是否好。
```

## 8. 你已经跑通的项目流程

你现在已经完成了一个非常重要的起点：

```text
1. 准备 Market1501 数据集
2. 使用 ResNet50 baseline
3. 加载 ImageNet 预训练权重
4. 训练 1 epoch
5. 测试 mAP / Rank-k
6. 生成 visrank 可视化结果
```

当前已经跑出的结果：

```text
mAP: 22.6%
Rank-1 : 40.6%
Rank-5 : 62.9%
Rank-10: 72.4%
Rank-20: 80.6%
```

这说明项目的数据读取、模型构建、训练、测试、结果展示都已经跑通。

## 9. 本课对应的项目关键文件

### 9.1 训练入口

```text
scripts/main.py
```

这是整个项目最重要的入口。

它负责：

```text
读取配置
构建 datamanager
构建 model
构建 optimizer
构建 engine
启动训练或测试
```

### 9.2 默认配置

```text
scripts/default_config.py
```

这里定义了项目默认参数，例如：

```text
data.root
data.sources
data.targets
train.max_epoch
train.batch_size
test.evaluate
test.visrank
```

### 9.3 ResNet50 配置

```text
configs/im_r50_softmax_256x128_amsgrad.yaml
```

这是你当前使用的 ResNet50 baseline 配置。

### 9.4 Market1501 数据读取

```text
torchreid/data/datasets/image/market1501.py
```

这个文件负责解析 Market1501 数据集。

重点关注：

```python
pattern = re.compile(r'([-\d]+)_c(\d)')
```

这行代码从图片文件名中解析：

```text
pid
camid
```

### 9.5 训练与测试引擎

```text
torchreid/engine/engine.py
torchreid/engine/image/softmax.py
```

这些文件负责训练循环、测试流程、特征提取、指标计算和结果展示。

### 9.6 结果可视化

```text
torchreid/utils/reidtools.py
```

`visrank` 生成的 top-k 检索图就是由这里的函数完成的。

## 10. 本课建议你动手做的事

### 任务 1：重新运行一次测试

使用已经训练好的权重，只做测试：

```bat
cd /d D:\code\deep-person-reid-master

F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py ^
  --config-file configs\im_r50_softmax_256x128_amsgrad.yaml ^
  --root D:\code\datasets ^
  data.workers 0 ^
  model.load_weights log\resnet50_market1501_pretrained_1epoch\model\model.pth.tar-1 ^
  test.evaluate True ^
  data.save_dir log\lesson01_test
```

### 任务 2：重新生成一次 visrank

```bat
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py ^
  --config-file configs\im_r50_softmax_256x128_amsgrad.yaml ^
  --root D:\code\datasets ^
  data.workers 0 ^
  model.load_weights log\resnet50_market1501_pretrained_1epoch\model\model.pth.tar-1 ^
  test.evaluate True ^
  test.visrank True ^
  test.visrank_topk 10 ^
  data.save_dir log\lesson01_visrank
```

### 任务 3：观察 10 张 visrank 图

打开下面目录：

```text
D:\code\deep-person-reid-master\log\lesson01_visrank\visrank_market1501
```

观察前 10 张图，尝试回答：

```text
1. 哪些 query 第一名找对了？
2. 哪些 query 第一名找错了？
3. 错误是因为衣服相似、姿态变化、遮挡，还是摄像头角度变化？
```

## 11. 本课小结

本课你需要记住：

```text
1. ReID 是检索任务，不只是分类任务。
2. ReID 数据集通常分为 train、query、gallery。
3. 训练时模型学习身份分类，测试时用特征向量做检索。
4. Rank-1 看第一名是否正确，mAP 看整体排序质量。
5. torchreid 的主入口是 scripts/main.py。
6. Market1501 的数据解析在 market1501.py。
7. visrank 是观察模型效果最直观的方式。
```

下一课建议学习：

```text
第 2 课：Market1501 数据集结构
```

