# 第 2 课：Market1501 数据集结构

## 1. 本课目标

学完本课，你需要能够回答下面几个问题：

```text
1. Market1501 数据集由哪些目录组成？
2. bounding_box_train、query、bounding_box_test 分别有什么作用？
3. 图片文件名里的 pid 和 camid 是什么？
4. torchreid 是如何从文件名中解析身份和摄像头编号的？
5. 为什么训练集需要 relabel，而 query/gallery 不需要？
```

本课的重点不是模型，而是数据。ReID 项目能不能跑通，第一关就是数据结构是否正确。

## 2. Market1501 是什么

Market1501 是 Person ReID 中非常经典的数据集。

它的任务是：

```text
给定一张 query 行人图片，
在 gallery 图库中找出同一个人的图片。
```

Market1501 已经把原始视频或图片中的行人检测框裁剪出来，因此你看到的不是完整街景，而是一张张行人 crop 图。

在 torchreid 中，Market1501 是默认支持的数据集之一。

项目关键文件：

```text
torchreid/data/datasets/image/market1501.py
```

## 3. 标准目录结构

在 torchreid 里，Market1501 推荐放成下面这种结构：

```text
D:\code\datasets\
└── market1501\
    └── Market-1501-v15.09.15\
        ├── bounding_box_train\
        ├── bounding_box_test\
        ├── query\
        ├── gt_bbox
        ├── gt_query
        └── readme.txt
```

其中最重要的是这三个目录：

```text
bounding_box_train
query
bounding_box_test
```

torchreid 真正训练和测试主要使用这三个目录。

你的当前项目中，我们通过目录联接让 torchreid 看到的结构是：

```text
D:\code\datasets\market1501\Market-1501-v15.09.15
```

也就是说，运行命令时 `--root` 应该传：

```text
D:\code\datasets
```

而不是：

```text
D:\code\datasets\market1501
```

## 4. 三个核心目录的作用

### 4.1 bounding_box_train

路径：

```text
Market-1501-v15.09.15\bounding_box_train
```

作用：

```text
训练模型
```

里面的图片带有身份编号。训练时，模型会把每个身份当成一个类别来学习。

在你之前跑通的 Market1501 中，torchreid 读取到：

```text
train ids    : 751
train images : 12936
cameras      : 6
```

也就是说，训练集中有 751 个身份，共 12936 张图片。

### 4.2 query

路径：

```text
Market-1501-v15.09.15\query
```

作用：

```text
测试阶段作为查询图片
```

每一张 query 图片都会被模型拿去 gallery 中检索。

在你之前跑通的 Market1501 中：

```text
query ids    : 750
query images : 3368
cameras      : 6
```

测试时，日志里这句：

```text
Extracting features from query set ...
Done, obtained 3368-by-512 matrix
```

意思是：

```text
query 一共有 3368 张图片
每张图片被提取成 512 维特征
```

所以得到矩阵：

```text
3368 x 512
```

### 4.3 bounding_box_test

路径：

```text
Market-1501-v15.09.15\bounding_box_test
```

作用：

```text
测试阶段作为 gallery 图库
```

模型会把每张 query 和 gallery 中所有图片计算距离，然后排序。

在你之前跑通的 Market1501 中：

```text
gallery ids    : 751
gallery images : 15913
cameras        : 6
```

测试日志里：

```text
Extracting features from gallery set ...
Done, obtained 15913-by-512 matrix
```

意思是：

```text
gallery 一共有 15913 张图片
每张图片被提取成 512 维特征
```

所以得到矩阵：

```text
15913 x 512
```

## 5. ReID 测试时 query 和 gallery 怎么配合

测试阶段的流程是：

```text
query 图片 -> 模型 -> query features
gallery 图片 -> 模型 -> gallery features
query features 和 gallery features -> 计算距离矩阵
距离从小到大排序 -> 得到 top-k 检索结果
```

如果 query 有 3368 张，gallery 有 15913 张，那么距离矩阵大小是：

```text
3368 x 15913
```

每一行表示：

```text
某一张 query 和所有 gallery 图片的距离
```

距离越小，模型认为越相似。

## 6. Market1501 图片文件名格式

Market1501 的图片文件名包含关键信息。

例如：

```text
0002_c1s1_000451_03.jpg
```

可以拆成：

```text
0002     -> person id，也就是身份编号
c1       -> camera id，也就是摄像头编号
s1       -> sequence 信息
000451   -> 帧或图片序号
03       -> 检测框编号或附加编号
```

torchreid 主要关心：

```text
pid
camid
```

也就是：

```text
0002 -> pid = 2
c1   -> camid = 1
```

## 7. pid 是什么

`pid` 是 person id 的缩写，也就是身份编号。

例如：

```text
0002_c1s1_000451_03.jpg
0002_c2s1_000301_01.jpg
0002_c5s1_001001_02.jpg
```

这些图片的 pid 都是：

```text
2
```

表示它们属于同一个人。

ReID 的训练目标之一就是让模型学会：

```text
同一个 pid 的图片特征更接近
不同 pid 的图片特征更远
```

## 8. camid 是什么

`camid` 是 camera id 的缩写，也就是摄像头编号。

例如：

```text
c1 -> 摄像头 1
c2 -> 摄像头 2
c3 -> 摄像头 3
```

Market1501 中通常有 6 个摄像头。

camid 在 ReID 中很重要，因为测试时通常会排除同摄像头下某些无效匹配，并且评价时强调跨摄像头检索能力。

## 9. torchreid 如何解析文件名

打开：

```text
torchreid/data/datasets/image/market1501.py
```

你会看到类似代码：

```python
pattern = re.compile(r'([-\d]+)_c(\d)')
```

这行正则表达式用于从图片路径中提取：

```text
pid
camid
```

核心逻辑大致是：

```python
pid, camid = map(int, pattern.search(img_path).groups())
```

对于：

```text
0002_c1s1_000451_03.jpg
```

解析结果就是：

```text
pid = 2
camid = 1
```

随后代码会做：

```python
camid -= 1
```

因为程序内部习惯从 0 开始编号：

```text
原始 c1 -> 内部 camid 0
原始 c2 -> 内部 camid 1
...
原始 c6 -> 内部 camid 5
```

## 10. 为什么训练集要 relabel

在 `market1501.py` 中，训练集读取时：

```python
train = self.process_dir(self.train_dir, relabel=True)
```

query 和 gallery 读取时：

```python
query = self.process_dir(self.query_dir, relabel=False)
gallery = self.process_dir(self.gallery_dir, relabel=False)
```

这说明：

```text
训练集需要 relabel
query/gallery 不需要 relabel
```

原因是训练分类器时，类别编号最好是连续的：

```text
0, 1, 2, 3, ..., num_classes - 1
```

但原始 pid 可能不是从 0 连续编号的。

所以训练集会把原始 pid 重新映射成连续标签。

例如：

```text
原始 pid: 2, 7, 15
训练 label: 0, 1, 2
```

这样分类器才能正常使用交叉熵 loss。

query/gallery 不 relabel，是因为测试评价需要使用原始身份编号判断检索是否正确。

## 11. junk pid 是什么

Market1501 中有一些特殊 pid：

```text
-1
0
```

在代码中：

```python
_junk_pids = [0, -1]
```

其中：

```text
-1 通常表示无效图片
0  通常表示背景或干扰项
```

读取时，代码会跳过 `pid == -1` 的图片：

```python
if pid == -1:
    continue
```

这些图片不作为正常身份参与训练。

## 12. 你之前测试日志中的数据统计

你之前成功运行后，日志里出现：

```text
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
```

这说明 torchreid 已经正确识别了数据集。

如果数据路径或结构不对，这里通常会报错，例如：

```text
RuntimeError: 'xxx' is not available
FileNotFoundError
```

所以看到这张统计表，就是一个很重要的信号：

```text
数据集读取成功。
```

## 13. 本课对应的项目关键文件

### 13.1 Market1501 数据类

```text
torchreid/data/datasets/image/market1501.py
```

重点看：

```python
dataset_dir = 'market1501'
self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
self.query_dir = osp.join(self.data_dir, 'query')
self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
```

以及：

```python
pattern = re.compile(r'([-\d]+)_c(\d)')
```

### 13.2 数据集基类

```text
torchreid/data/datasets/dataset.py
```

这个文件包含数据集检查、下载、统计等通用逻辑。

### 13.3 数据集注册入口

```text
torchreid/data/datasets/__init__.py
```

这个文件负责根据名字找到对应数据集类。

例如：

```text
market1501 -> Market1501
```

### 13.4 DataManager

```text
torchreid/data/datamanager.py
```

DataManager 会调用数据集类，并把数据转换成 DataLoader。

## 14. 本课动手任务

### 任务 1：查看 Market1501 目录

在 PowerShell 中执行：

```powershell
Get-ChildItem D:\code\datasets\market1501\Market-1501-v15.09.15
```

确认存在：

```text
bounding_box_train
query
bounding_box_test
```

### 任务 2：查看几张训练图片文件名

```powershell
Get-ChildItem D:\code\datasets\market1501\Market-1501-v15.09.15\bounding_box_train | Select-Object -First 10
```

观察文件名中的：

```text
pid
camid
```

### 任务 3：重新跑一次只测试命令

```powershell
cd D:\code\deep-person-reid-master

F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py --config-file configs\im_r50_softmax_256x128_amsgrad.yaml --root D:\code\datasets data.workers 0 model.load_weights log\resnet50_market1501_pretrained_1epoch\model\model.pth.tar-1 test.evaluate True test.batch_size 64 data.save_dir log\lesson02_test
```

重点观察日志中的数据统计表：

```text
train
query
gallery
```

## 15. 和自定义老虎数据集的关系

以后你做老虎 ReID，也需要类似结构。

你可以设计成：

```text
tigerreid
├── bounding_box_train
├── query
└── bounding_box_test
```

或者自己写一个 Dataset 类，读取你自己的目录结构。

但无论如何，每张图片最终都需要提供：

```text
img_path
pid
camid
```

这三个信息。

对于老虎数据：

```text
pid   -> 老虎个体编号
camid -> 拍摄地点、摄像机、视频源或人为设定的域编号
```

如果暂时没有真实摄像头编号，也可以先统一设为 0，但更规范的做法是根据视频源或采集点区分 camid。

## 16. 本课小结

本课你需要记住：

```text
1. Market1501 的核心目录是 bounding_box_train、query、bounding_box_test。
2. train 用于训练，query 用于查询，gallery 用于被检索。
3. 图片文件名中包含 pid 和 camid。
4. torchreid 通过正则表达式从文件名中解析 pid 和 camid。
5. 训练集 relabel 是为了让分类标签连续。
6. query/gallery 不 relabel，是为了保留原始身份用于评价。
7. 自定义数据集最终也要提供 img_path、pid、camid。
```

下一课建议学习：

```text
第 3 课：torchreid 项目结构
```

