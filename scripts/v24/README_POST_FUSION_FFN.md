# CMFF受控差分：每轮融合后4倍FFN对照

当前主线按用户决定继续固定为四轮CMFF共享八专家、原生动态Top-2、±0.1条件门＋RMS幅度匹配差分。本批仅新增两次TaxiBJ random0.4训练，不把FFN预先确立为第三项贡献。

| 组 | 改动 | 对照 |
| --- | --- | --- |
| F01 | 四轮共享一个FFN | 已完成K04 seed7：有/无额外FFN |
| F02 | 四轮各有独立FFN，共四个 | F01：仅改变FFN共享；K04：有/无FFN |

每轮在原生Top-2专家加权融合并恢复到细网格之后执行：

`Hnext = U + Conv1x1(256→64)(GELU(Conv1x1(64→256)(LayerNorm(U))))`

U为融合后状态；输入[B,64,12,32,32]，LayerNorm逐位置归一化通道，两个1×1×1 Conv3d等价于逐位置Linear。采用Pre-LN残差，无dropout，无额外门或残差缩放，线性层默认非零初始化。每轮专家更新仍为direct；FFN内部加入其自身残差。FFN之后再解码及传入下一轮，所以后续受控差分作用于包含FFN的状态，Hprev仍为上一轮输入。FFN共享与专家共享是两个独立开关，本批专家池始终共享。

新增模块隔离seed+94001随机流。F02四个FFN由F01同一初始FFN深拷贝、随后独立训练；两组初始函数相同，公共骨干逐值核对并加载旧K04 seed7初始化。F01新增33216参数，F02新增132864参数；两组均额外执行四次全分辨率FFN。原专家面积代理4.625不包含FFN成本；FFN主线性层额外约1.611G MAC/窗口，不含bias/LN/GELU。

共同协议直接来自冻结K04：seed7、100epoch、batch32、每5epoch验证、GPU0单卡串行、AdamW lr1e-3余弦至3e-4、wd1e-4、clip1、AMP、严格确定性、无早停、L1＋0.01candidate均衡。训练四基础mask族近似均衡、rate0.4、逐epoch重采样；mask种子20260917，val+20000/test+30000。没有额外视图、一致性或自由尺度。

保存best.pth与last.pth，best仅按ID验证MAE；原六套测试全部运行。旧K04来自`outputs/v24-COE/experiments/core_validation/684cad05994af9a4`的`taxibj_K04_seed7`，复制其结果、配置、初始化和日志作冻结参考，不重训。不同日期的时间测量只作实测参考，不视作严格控制硬件时钟的性能实验。

```bash
conda activate project
python -m unittest discover -s tests -p test_v24_post_fusion_ffn.py
CUDA_VISIBLE_DEVICES=0 python scripts/v24/preflight_post_fusion_ffn.py --output /tmp/post_ffn_preflight.json
python -u scripts/v24/run_post_fusion_ffn.py --gpu 0 --preflight /tmp/post_ffn_preflight.json
# 恢复/补齐只使用冻结suite；已完成训练跳过，缺评估单独补齐：
python -u scripts/v24/run_post_fusion_ffn.py --gpu 0 --suite <suite>
```

队列冻结源码、数据指纹、K04参考和新增初始化；失败立即停，不自动减batch或转GPU1。断点完整恢复由原train_four_direction实现。每组完成后自动导出阶段报告；两组训练/六套评估全部完成后按项目约定导出正式报告到`experments_report/`，顺序为方法复现→完整结果/曲线/诊断→分析/局限/建议，保留ID/OOD退步和成本。初步≥1% ID改善只算单seed信号，不修改原主线证据边界。
