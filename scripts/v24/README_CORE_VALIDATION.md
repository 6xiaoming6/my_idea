# 前两个贡献的固定主干验证

本批固定研究 CMFF 多尺度共享专家链及 ±0.1 受控差分，第三贡献暂缓；无额外视图、一致性、自由尺度路由、软预热或动态选候选。

## 方法与训练次数

| 方法 | 路径 | 专家池 | 差分 | 对照 |
| --- | --- | --- | --- | --- |
| K01 | FFFF | 共享 | 无 | 单尺度参考 |
| K02 | CMFF | 共享 | 无 | K01：多尺度 |
| K03 | FFFF | 共享 | 条件门＋RMS匹配 | K01：单尺度下的差分 |
| K04 | CMFF | 共享 | 条件门＋RMS匹配 | K02：多尺度下的差分；K03：有差分时的多尺度 |
| K05 | CMFF | 独立 | 无 | K02：无差分时共享/独立 |
| K06 | CMFF | 独立 | 条件门＋RMS匹配 | K04：有差分时共享/独立；K05：独立池差分 |
| K07 | CMFF | 共享 | 门输入固定零，保留原网络/RMS | K04：条件化是否必要 |
| K08 | CMFF | 共享 | 条件门＋原始差分，无RMS匹配 | K04：幅度匹配是否必要 |

K01–K04 在 TaxiBJ、BikeNYC 上分别使用 seed7/17/27，24次；K05–K08 仅 TaxiBJ seed7，4次。默认共28次。仅传 `--dataset taxibj` 为16次。先运行TaxiBJ seed7八组，再seed17/27四组，之后BikeNYC三种子四组。旧结果不复用为本批匹配参考。

## 共同协议

100epoch，batch32，每5epoch验证，GPU0单卡串行，AdamW，lr1e-3余弦至3e-4，weight_decay1e-4，clip1，AMP，无早停。保存best.pth与last.pth，best只按ID验证MAE。模型/loader同种子，mask种子固定20260917，原时间切分和split偏移，四基础缺失族random0.4训练逐epoch重采样，验证固定。

四轮八专家T/S/TD/SD/TA/ST/TL/SL，原生Top2、组内softmax、direct；无completion feedback、接受门、搭档路由、路由输入噪声及软启动。L1＋0.01 candidate均衡。所有尺度保持时间长度12。TaxiBJ F/M/C=32²/16²/8²；BikeNYC=24×12/12×6/6×3。

差分：当前轮输入H减上一轮输入Hprev；按样本通道在T/H/W上计算RMS并匹配H幅度，比例detach、分母下限1e-6。门输入为H和Hprev各自全局/缺失均值摘要；共享MLP输出0.1*tanh，最后一层零初始化。只修改专家输入，发生在尺度池化、投影与归一化之前。首轮无历史，历史不跨batch，Router读取原H。K07将门网络输入置零但不更换网络；一部分参数梯度恒零，总可训参数不等于有效容量。

公共参数从同seed模板逐值核验并加载，新增模块隔离随机流；独立池由共享初始池复制后独立训练，不改变Router/编码器初始化。所有模型保留同一冻结尺度头，无差分组冻结未使用的门头。报告总参数/可训参数，不宣称严格等参数。

严格确定性，关闭TF32、benchmark；沿用W系列确定性池化和插值。K04与W01模型前向/梯度/状态字典兼容。原W/U/R/T默认路径不变。每窗口8次专家调用；FFFF面积代理8、CMFF面积代理4.625，不等于相同FLOPs。

## 启动、验收及恢复

```bash
python -u scripts/v24/run_core_validation.py --gpu 0 --epochs 100 --batch-size 32
python -m unittest discover -s tests -p test_v24_core_validation.py
CUDA_VISIBLE_DEVICES=0 python scripts/v24/preflight_core_validation.py --output /tmp/core_preflight.json
```

入口支持 `--dataset`、`--seeds`、`--variants`、`--dry-run`、`--summary-only`、`--suite`。指定suite时以冻结计划为准；冻结源代码、配置、数据指纹与每组初始化；完成训练跳过、缺评估单独补跑，中断从last恢复全部训练状态。失败立即停止；不降低batch、不启用GPU1。

控制台仅任务切换及 train epoch 当前/总数 batch级tqdm（train loss/mae/rmse）。详细诊断按已有压缩日志策略写入文件。

## 评估与结论边界

全部运行原六套测试（ID、两元、形态、三元、两元mask复测、形态mask复测），种子不变。逐数据集逐训练seed配对MAE/RMSE，三种子报告均值与样本标准差、逐seed差值；mask复测不视为训练seed复验，不把两数据集原始MAE直接平均。

重点预先规定：K02/K01和K04/K03检验尺度；K03/K01和K04/K02检验差分；比较两种路径下差分收益，分析是否依赖多尺度。K05–K08只有单种子信号，不能写成稳定结论。共享结构可贡献参数/成本收益，不要求其必然优于更大独立模型。若K07与K04相当，不能把状态条件化作为已证实优势。

先看ID三个种子的改善方向，平均相对改善≥1%作为值得推进的实际收益信号，仍报告全部差值；OOD各协议单列，退步>5%显式标记。固定预选方法，不基于测试挑候选。现有B系列提供多轮背景，本批没有新增单层Top8对照，不能仅凭本批宣称多轮优于单层或CMFF最优。

每个数据集/seed分块结束导出阶段报告；完成28组后自动正式报告到experments_report，顺序为方法与复现→完整结果/曲线/诊断→分析/局限/建议。不得因结果不好更换结论门槛或隐去方法。
