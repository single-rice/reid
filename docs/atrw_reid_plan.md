# ATRW 自定义数据集重识别接入说明

## 当前数据

```text
reid-data/ATRW/
├── atrw_reid_train/train/
└── atrw_anno_reid_train/reid_list_train.csv
```

`reid_list_train.csv` 格式：

```text
tiger_id,image_name
```

当前检查结果：

```text
CSV 标注图片 : 1887
身份数       : 107
每个身份图片 : 10 到 98 张
CSV 缺失图片 : 0
图片目录总数 : 3392
未标注图片   : 1505
```

本次只使用 CSV 中有身份标注的 1887 张图片。

## 划分策略

按身份划分，避免同一只老虎同时出现在训练集和测试集：

```text
train : 排序后的前 60% 身份
test  : 剩余 40% 身份
```

测试集内部：

```text
query   : 每个测试身份取排序后的第 1 张图片
gallery : 同身份剩余图片
```

当前划分后：

```text
train   : 64 个身份，1087 张图片
query   : 43 个身份，43 张图片，每个身份 1 张
gallery : 43 个身份，757 张图片
```

由于当前没有真实 camid，沿用最小可用策略：

```text
train   : camid = 0
query   : camid = 0
gallery : camid = 1
```

## 已新增文件

```text
torchreid/data/datasets/image/atrw.py
configs/im_r50_softmax_256x128_atrw.yaml
```

并在以下入口注册：

```text
torchreid/data/datasets/image/__init__.py
torchreid/data/datasets/__init__.py
```

## 训练命令

```powershell
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py `
  --config-file configs\im_r50_softmax_256x128_atrw.yaml `
  --root reid-data
```

## 只评估命令

```powershell
F:\Users\ROG\anaconda3\envs\xiaotudui\python.exe scripts\main.py `
  --config-file configs\im_r50_softmax_256x128_atrw.yaml `
  --root reid-data `
  model.load_weights log\resnet50_atrw_softmax\model\model.pth.tar-60 `
  test.evaluate True
```
