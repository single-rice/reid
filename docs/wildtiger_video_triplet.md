# WildTiger 跨视频训练

重新运行 `tools/prepare_wildtiger.py`，输出到一个新的目录，生成含
`pid,image,camid,video_id` 的 train.csv 和 val.csv。camid 是同一身份内的视频编号，
由该身份全部训练和验证视频统一映射；不是物理相机编号。测试协议不变。
旧 CSV 仍可加载，但缺少 camid 时默认为 0，无法提供跨视频监督。

启动：

```bash
python scripts/main.py --config-file configs/wildtiger_video_triplet.yaml --root /path/to/reid-data
```

每批 8 个身份，每身份 8 张；多视频身份随机选两个视频，分别抽 4 张。
视频不足 4 张时补采；单视频身份沿用普通身份采样。
`sampler.video_batch_probability=0.5` 表示每批以 50% 概率优先留一个身份位置给
尚有配额的多视频身份，其余位置正常抽取。不会重新填充身份配额，所以不保证
整轮恰好 50% 批次包含多视频身份。与原采样器一样，不足完整批次的剩余身份会丢弃，
`len(sampler)` 是样本数上界估计。

总损失是 `0.5 * loss_t + 1.0 * loss_cv + 1.0 * loss_x`。
跨视频损失只平均有跨视频正样本和不同身份负样本的 anchor；没有有效 anchor 时返回
可反向传播的零。日志 `cross_video_anchor_fraction` 可检查实际监督覆盖率。
默认 weight_cv=0，保持原训练配置行为。

建议固定数据划分和种子，比较普通采样、仅视频采样、视频采样加跨视频损失三组。
观察 multi 查询组指标和覆盖率；损失下降本身不能证明跨视频泛化改善。
这里没有重新统计实际数据，29/323 的身份分布仍需在训练数据上确认。
