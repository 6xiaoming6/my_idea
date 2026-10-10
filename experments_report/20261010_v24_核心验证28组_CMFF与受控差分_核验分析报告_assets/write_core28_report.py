from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, statistics as st, math, gzip, shutil, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/v24_core28_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/home/students/HuangMingYu/code/py/my_idea/my_idea')
SUITE=ROOT/'outputs/v24-COE/experiments/core_validation/684cad05994af9a4'
INPUT=Path('/tmp/v24_core28_audit')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
plan=read(SUITE/'plan.json');rows=read(INPUT/'rows.json');curves=read(INPUT/'curves.json');evals=read(INPUT/'evaluations.json');verification=read(INPUT/'verification.json')
assert len(rows)==28 and verification['completed']['count']==28
assert not verification['source_mismatch'] and not verification['failures']
assert not verification['current_source_difference']
assert all(not x['issues'] for x in rows.values())
assert all(x['sha256_match'] and x['stat_match'] for x in verification['data'].values())
DATE='20261010'
DEST=ROOT/'experments_report'/f'{DATE}_v24_核心验证28组_CMFF与受控差分_核验分析报告.md'
ASSETS=DEST.with_name(DEST.stem+'_assets');ASSETS.mkdir(exist_ok=True)
shutil.copyfile(INPUT/'verification.json',ASSETS/'verification.json');shutil.copyfile(INPUT/'rows.json',ASSETS/'results_summary.json')
shutil.copyfile('/tmp/audit_v24_core28.py',ASSETS/'audit_core28.py')
shutil.copyfile('/tmp/write_v24_core28_report.py',ASSETS/'write_core28_report.py')
with gzip.open(ASSETS/'full_evaluations.json.gz','wt',encoding='utf-8') as f:json.dump(evals,f,ensure_ascii=False,separators=(',',':'))
compact={n:[{**{k:x[k] for k in ['epoch','is_best','perf']},**{s:({k:v for k,v in x[s].items() if k in ['loss','mae','rmse','lr','train_optimizer_steps','train_skipped_amp_steps'] or 'memory_gate' in k or k in ['coe_pair_path_unique_count','coe_pair_path_max_fraction']} if x.get(s) else None) for s in ['train','val']}} for x in h] for n,h in curves.items()}
write(ASSETS/'curves.json',compact)
DS={'taxibj':'TaxiBJ','bikenyc':'BikeNYC'}
SETS={'in_distribution':'ID','unseen_combinations':'两元组合','unseen_geometry':'未见形态','unseen_triple':'三元组合','unseen_combinations_repeat':'两元 mask 复测','unseen_geometry_repeat':'形态 mask 复测'}
METHODS={
 'K01':('FFFF；共享池；无差分','K01','单尺度参考'),
 'K02':('CMFF；共享池；无差分','K01','无差分时的多尺度收益'),
 'K03':('FFFF；共享池；±0.1 条件门＋RMS 匹配差分','K01','FFFF 下差分的独立贡献'),
 'K04':('CMFF；共享池；±0.1 条件门＋RMS 匹配差分','K02 / K03','差分与多尺度的独立贡献及交互'),
 'K05':('CMFF；四轮独立池；无差分','K02','无差分时共享的精度/成本取舍'),
 'K06':('CMFF；四轮独立池；条件门＋RMS 差分','K04 / K05','有差分时共享取舍及独立池中的差分'),
 'K07':('CMFF；共享池；原门 MLP 输入置零；保留 RMS','K04','门的状态条件化是否必要'),
 'K08':('CMFF；共享池；条件门传递原始差分；无 RMS 匹配','K04','RMS 幅度匹配是否必要')}
PAIRS=[('K02','K01','无差分尺度'),('K04','K03','有差分尺度'),('K03','K01','FFFF 差分'),('K04','K02','CMFF 差分')]
ABLATIONS=[('K05','K02','无差分独立池/共享池'),('K06','K04','有差分独立池/共享池'),('K06','K05','独立池差分'),('K07','K04','无条件/条件门'),('K08','K04','无 RMS/有 RMS')]
def metric(d,k,z,q,m):return rows[f'{d}_{k}_seed{z}']['test'][q][m]
def group(d,k):return [rows[f'{d}_{k}_seed{z}'] for z in [7,17,27]]
def change(d,a,b,z,q,m):return 100*(metric(d,a,z,q,m)/metric(d,b,z,q,m)-1)
def fmt(x):return f'{x:.4f}'
def mean_sd(a):return f'{st.mean(a):.4f} ± {st.stdev(a):.4f}'
def pct(x):return f'{x:+.2f}%'+(' **†**' if x>5 else '')
def link(p,label):return f'[{label}]({p})'
def asset(name,label):return link(ASSETS.name+'/'+name,label)

lines=[f'# {DATE}_v24_核心验证28组_CMFF与受控差分_核验分析报告','',
 f'生成日期：2026-10-10（中国时间）；批次：`684cad05994af9a4`；代码分支：`v24-COE`，HEAD：`{verification["git_head"]}`。历史执行来源以冻结源码和逐组保存配置为准。',
 '', '## 1. 实验具体做法与复现信息','','### 1.1 全部训练的做法与对照','','| 实验 ID | 数据集 / 训练种子 | 具体做法 | 对照 | 验证问题 |','|---|---|---|---|---|']
for n,j in plan['jobs'].items():
 desc,ref,question=METHODS[j['method']]
 lines.append(f'| {n} | {DS[j["dataset"]]} / {j["config"]["seed"]} | {desc} | 同数据集、同种子的 {ref} | {question} |')
lines += ['', 'K01–K04：两个数据集 × 三个独立训练种子（7/17/27）× 四种方法，共 24 次。K05–K08：仅 TaxiBJ seed7，共 4 次。所有组均从头训练；六套测试是同一个训练模型的评估，不增加训练重复数。',
 '', '### 1.2 共同模型与状态流',
 '', '输入为 `[B,2,12,H,W]` 的时空流量和观测 mask；隐藏通道为 64。mask=1 表示可见，隐藏输入置零。编码器读取可见值、mask、仅由 mask 计算的十类观测支持特征和位置编码。初始 completion 在可见位置取原值、缺失位置取初始解码预测。四轮中该 completion 与观测 mask 保持原定义；轮次预测不会变成新的观测，本批关闭 completion feedback。',
 '', '每轮有独立 Router：由当前原始隐藏状态 H、初始 completion 和支持摘要等信息生成八专家概率，动态选择原生 Top-2，只对被选择的样本/专家执行分发，组内 softmax 混合。尺度则事先固定为 FFFF 或 CMFF；本批没有动态尺度选择。CMFF 的 C/M 轮将输入在空间维确定性平均池化、执行专家后上采样回细网格，时间长度始终为 12。direct 更新把混合输出作为下一轮 H；不是新增跨专家轮的恒等残差块，也没有每轮后新增通用 FFN。最后一轮解码得到最终输出。',
 '', '| 专家 | 实际算子 |','|---|---|',
 '| T / S | 时间 / 空间深度卷积；核宽 3；通道 64→128→64，含 GELU 和逐位置 LayerNorm |',
 '| TD / SD | 时间 / 空间膨胀深度卷积；核宽 3、dilation=2 |',
 '| TA | 每个空间位置独立执行长度 12 的双向时间注意力，4 heads；内部已有 2 倍通道 FFN |',
 '| ST | 联合时空 3×3×3 深度卷积 |',
 '| TL / SL | 时间 / 空间较大核深度卷积；核宽 5 |',
 '', '| 数据集 | F / M / C 网格 | train / val / test 窗口数 | 每 epoch train batch 数 |', '|---|---|---|---|']
for d in DS:
 r=group(d,'K01')[0];grid='32×32 / 16×16 / 8×8' if d=='taxibj' else '24×12 / 12×6 / 6×3'
 lines.append(f'| {DS[d]} | {grid} | {r["samples"]["train"]} / {r["samples"]["val"]} / {r["samples"]["test"]} | {r["steps"]["train"]} |')
lines += ['', '### 1.3 受控差分与逐组实现',
 '', '有历史的轮次取 D=H−Hprev，其中 Hprev 是上一轮输入，首轮不传历史，历史不跨 batch。对每个样本和通道，在 T/H/W 上计算 RMS，令匹配消息为：',
 '', r'\[\widetilde D=D\cdot\operatorname{stopgrad}\left(\frac{\operatorname{RMS}(H)}{\max(\operatorname{RMS}(D),10^{-6})}\right),\qquad g=0.1\tanh\bigl(\operatorname{MLP}([s(H),s(H_{prev})])\bigr),\qquad H_{expert}=H+g\odot\widetilde D.\]',
 '', 's(H) 拼接全局均值和缺失位置加权均值，两个状态组成 4×64 维门输入。门 MLP 在各轮共享，末层零初始化；仅 RMS 缩放比 detach，差分内容及门特征不 detach。g 为每样本每通道有符号系数，广播到时空位置。±0.1 约束的是系数；RMS 匹配使消息幅度与 H 对齐（低于分母下限时受限），并不约束每个位置的绝对值，也不保证预测误差下降。修正位于尺度池化、公共投影和归一化之前。Router 读取原 H，后续轮会间接受到新状态的影响。',
 '', '- **K01 / K02**：只差固定尺度；关闭差分，保留但冻结未使用门网络。',
 '- **K03 / K04**：分别在 FFFF / CMFF 上启用上述完整差分，其余训练设置保持一致。',
 '- **K05 / K06**：第一轮保留原专家池，后三轮从共享初始池逐值复制，之后独立训练，共四个八专家池；Router、编码器及其他公共模块初始化不变。',
 '- **K07**：使用相同门网络，将整个门输入置零；保留可学习参数及 RMS。门因而不随样本状态变化，是各轮共享的通道常数，但不同通道可不同。部分参数梯度恒零，名义可训练参数数目不等于有效容量。',
 '- **K08**：只取消 RMS 匹配，使用 H+g⊙D；门的输入、±0.1 bound、零初始化和梯度路径不变。',
 '', '### 1.4 训练、缺失与评估协议',
 '', '统一 100 epoch、batch32（尾 batch 保留）、num_workers=0，GPU0 单卡串行。AdamW，lr=1e−3，余弦到第 100 epoch 的 3e−4，weight_decay=1e−4，clip=1，AMP，无早停。损失为缺失目标上的 L1＋0.01 candidate 均衡，关闭中间轮监督与 z-loss。每 5 epoch 验证，只按 ID 验证 MAE 选择 best.pth；last.pth 保存优化器、scheduler、AMP scaler、RNG 和恢复状态。训练 loss 含均衡项且有 batch 聚合口径，不能等同于全数据集缺失点 MAE。',
 '', '模型与 loader 使用训练 seed7/17/27；缺失生成 RNG 使用固定 mask seed，与训练 seed 独立。每个窗口安排一个基础族、缺失率 0.4，族分配接近平衡；训练逐 epoch 重排和重采样，验证/测试固定。基础族为随机点、节点中断、连续时间缺口、空间区域。mask 按 round(T×H×W×0.4) 精确缺失数量构造，因此网格舍入后的实际率可略偏离 0.4。mask 家族标签只用于诊断，不输入模型。',
 '', '训练 mask 基础 seed 为 20260917；验证有效 seed=20260917+20000=20280917；测试使用以下基础 seed 再加 30000。未见组合由基础族几何合成，两个分量随机分配约 40%–60% 新缺失位置；三元分量分配约三分之一新缺失位置。未见形态为移动区域和多块缺失。训练没有新增第二视图、组合族或一致性损失。',
 '', '| 协议 | 缺失家族 | 基础 mask seed | 测试有效 seed |','|---|---|---|---|']
protocol=next(iter(plan['jobs'].values()))['config']['experiment_plan']['protocol']
for k,v in protocol['evaluations'].items():lines.append(f'| {SETS[k]} (`{k}`) | {", ".join(v["families"])} | {v["seed"]} | {v["seed"]+30000} |')
lines += ['', '六套评估使用同一测试时间切分、相同窗口；每个协议内所有方法和训练 seed 使用相同固定 mask。MAE/RMSE 对所有有效缺失点累积误差和计数；分族均值不能未经计数加权替代总体 MAE。MAPE 对零/近零流量不稳定，本报告以 MAE/RMSE 为主。两个数据集分别训练，不能解释为跨数据集零样本迁移。',
 '', '### 1.5 公平性、环境与数据来源',
 '', '公共参数来自同 seed 冻结模板，新增模块隔离随机流；每个 seed 内包含两数据集的公共初始化 hash 一致。独立池复制后训练；保留同一冻结尺度头。报告总参数及可训练参数，不声称严格等有效容量。所有方法每窗口四轮×Top-2=8 次专家调用；FFFF 面积代理=8，CMFF=4.625，减少 42.1875%。面积代理不是 FLOPs：八种专家算子成本不同，动态专家选择也会改变实际耗时。',
 '', '执行环境：Python 3.9.25、PyTorch 2.6.0+cu124、CUDA 12.4、cuDNN 90100、RTX 3090。严格确定性开启，TF32 和 cuDNN benchmark 关闭，CUBLAS_WORKSPACE_CONFIG=:4096:8，使用确定性池化/插值及 math SDPA。AMP 仍存在少量 scaler 跳步，所以相同 epoch 不等于完全相同的有效优化器步数；详见逐组计数。',
 '', f'冻结批次：{link(SUITE/"plan.json","plan.json")}；{link(SUITE/"resolved_jobs.json","resolved_jobs.json")}；{link(SUITE/"initialization_audit.json","初始化审计")}；{link(SUITE/"source_snapshot","冻结源码")}。当前检查时，431 个清单文件全部与冻结副本逐文件一致；Git 存在已有未提交修改，HEAD 不能单独重建此次运行。完整 Git 状态和源码/数据校验见 {asset("verification.json","verification.json")}。',
 '', '实际入口与模型继承：冻结 `scripts/v24/run_core_validation.py` → `train_four_direction.py`；`CoreValidationCoE` → `BackboneExplorationCoE` → `IDPriorityCoE` → `FourDirectionCoE`。核心模型、门控、直接更新和评估口径均从冻结源码核对，不将早期 U/W 探索或本轮之外的设想写成已执行方法。',
 '', '原始启动参数（仅作复现记录，本次没有执行）：', '', '```bash',
 '/home/students/HuangMingYu/anaconda3/envs/difftdi/bin/python -u scripts/v24/run_core_validation.py --gpu 0 --epochs 100 --batch-size 32',
 f'# 指定既有冻结批次时，程序按冻结计划跳过已完成训练：\n/home/students/HuangMingYu/anaconda3/envs/difftdi/bin/python -u scripts/v24/run_core_validation.py --suite {SUITE} --gpu 0', '```',
 '', '原启动命令只有在工作区与冻结源清单一致时才复现本批；逐组实际配置及初始化路径保存在 resolved_jobs。以下映射给出全部运行证据：',
 '', '| 实验 | 配置 | 训练输出 / 原始日志 | 评估回执 |','|---|---|---|---|']
for n,r in rows.items():lines.append(f'| {n} | {link(SUITE/"configs"/(n+".json"),"配置")} | {link(Path(r["run_dir"]),"run")} / {link(Path(r["run_dir"])/"logs/metrics.jsonl","metrics.jsonl")} | {link(SUITE/"evaluations"/(n+".json"),"六套测试")} |')
lines += ['', f'独立核验脚本：{asset("audit_core28.py","audit_core28.py")}，只读取实验、输出核验数据到 `/tmp/v24_core28_audit/`，不训练/补评估。报告生成脚本：{asset("write_core28_report.py","write_core28_report.py")}；先以 difftdi 环境运行审计脚本，再运行生成脚本即可重建表格和图。原队列已自动导出的完整分析另存为 {link(ROOT/"experments_report/20261010_v24_前两贡献主干验证_684cad05994af9a4_完整分析.md","自动完整报告")}，本核验报告保留该文件。',
 '', '## 2. 完整结果、训练曲线与诊断',
 '', '### 2.1 完成性及检查点核验',
 '', '完成时间：2026-10-10 09:53:25（中国时间）。28/28 训练完成、28/28 best/last 检查点齐全、168/168 测试协议完整；没有失败回执、缺失组或主指标 NaN/Inf。宿主机只读检查时已无本批队列/训练进程。逐组检查 100 条连续 epoch、20 次验证、最佳 epoch 与最佳 MAE、检查点内配置、best 模型与 last 中缓存 best 的状态 hash、评估 checkpoint/协议、原始 ID test 与回执均一致。431 个冻结源码和六个 NPZ 数据指纹均匹配；28 个初始化模板校验通过，独立池初始复制逐值一致。',
 '', '### 2.2 全部组的训练与 ID 测试',
 '', '| 实验 | epoch / best epoch | best val MAE | ID MAE | ID RMSE | 优化器更新数 | AMP 跳步 |','|---|---|---|---|---|---|---|']
for n,r in rows.items():lines.append(f'| {n} | 100 / {r["best_epoch"]} | {fmt(r["best_val_mae"])} | {fmt(r["test"]["in_distribution"]["mae"])} | {fmt(r["test"]["in_distribution"]["rmse"])} | {int(r["optimizer_steps"])} | {int(r["amp_skips"])} |')
lines += ['', 'TaxiBJ 每组 7700 个训练 batch，BikeNYC 每组 1600 个。AMP 跳步范围 4–13 次，没有空监督 batch 跳过；这些是日志中实际记录的 scaler 行为，不计为训练任务失败，也不隐去有效更新次数差异。',
 '', '### 2.3 三训练种子汇总：各协议 MAE / RMSE',
 '', '均值 ± 样本标准差（n=3、ddof=1）。K05–K08 不混入此汇总；两套 mask 复测单列，不视为新增训练 seed。',
 '', '| 数据集 | 方法 | 测试协议 | MAE 均值 ± SD | RMSE 均值 ± SD |','|---|---|---|---|---|']
for d in DS:
 for k in ['K01','K02','K03','K04']:
  for q in SETS:lines.append(f'| {DS[d]} | {k} | {SETS[q]} | {mean_sd([metric(d,k,z,q,"mae") for z in [7,17,27]])} | {mean_sd([metric(d,k,z,q,"rmse") for z in [7,17,27]])} |')
lines += ['', '### 2.4 全部六套测试的逐组结果', '', '| 实验 | 协议 | MAE | RMSE |','|---|---|---|---|']
for n,r in rows.items():
 for q,m in r['test'].items():lines.append(f'| {n} | {SETS[q]} | {fmt(m["mae"])} | {fmt(m["rmse"])} |')
lines += ['', '### 2.5 全部缺失家族结果', '', '下表含 ID 四基础族、OOD 两元/形态/三元及其 mask 复测，原始计数也披露。', '', '| 实验 | 协议 | 缺失家族 | 有效缺失点数 | MAE | RMSE |','|---|---|---|---|---|---|']
families=[]
for n,e in evals.items():
 for q,a in e['sets'].items():
  m=a['metrics']
  for fam in protocol['evaluations'][q]['families']:
   pre='coe_family_'+fam+'_';record={'name':n,'set':q,'family':fam,'count':m[pre+'count'],'mae':m[pre+'mae'],'rmse':m[pre+'rmse']};families.append(record)
   lines.append(f'| {n} | {SETS[q]} | {fam} | {int(record["count"])} | {fmt(record["mae"])} | {fmt(record["rmse"])} |')
write(ASSETS/'family_metrics.json',families)
lines += ['', '### 2.6 实测成本',
 '', '分钟列为训练回执中逐 epoch 训练＋验证计时之和，不含模型创建、检查点落盘、六套测试及队列空隙。显存为该组训练各 epoch 的 max_memory_allocated 峰值（GiB），不是 nvidia-smi 显存占用。前向耗时读取 ID 测试内测量的 ms/sample/rank；是本次运行实测，不是独立多次预热、固定专家路径的性能基准。',
 '', '| 实验 | 总参数 | 可训参数 | 训练＋验证分钟 | 峰值 GiB | ID 前向 ms/窗口 | 调用/窗口 | 面积代理 |','|---|---|---|---|---|---|---|---|']
for n,r in rows.items():
 m=evals[n]['sets']['in_distribution']['metrics'];lines.append(f'| {n} | {r["parameters"]} | {r["trainable"]} | {r["minutes"]:.2f} | {r["peak_gib"]:.3f} | {m["forward_ms_per_sample_per_rank"]:.3f} | {m["coe_expert_execution_count"]:.0f} | {m["coe_expert_grid_equivalents"]:.3f} |')
lines += ['', '| 数据集 | 方法 | 分钟均值 ± SD | 峰值 GiB 均值 ± SD | 前向 ms/窗口均值 ± SD |','|---|---|---|---|---|']
for d in DS:
 for k in ['K01','K02','K03','K04']:
  g=group(d,k);lines.append(f'| {DS[d]} | {k} | {mean_sd([x["minutes"] for x in g])} | {mean_sd([x["peak_gib"] for x in g])} | {mean_sd([evals[x["name"]]["sets"]["in_distribution"]["metrics"]["forward_ms_per_sample_per_rank"] for x in g])} |')
lines += ['', '### 2.7 训练曲线与收敛记录',
 '', '每个数据集/训练 seed 一张图，包含训练 loss、训练 MAE、ID 验证 MAE 与 ID 验证 RMSE；验证每 5 epoch 一点，无插值补齐未验证 epoch。TaxiBJ seed7 图包含八组，其余图四组。',
 '', '| 实验 | train loss：epoch1→100 | train MAE：epoch1→100 | val MAE：epoch5→100 | last val−best val |','|---|---|---|---|---|']
for n,r in rows.items():
 h=curves[n];lines.append(f'| {n} | {h[0]["train"]["loss"]:.3f}→{h[-1]["train"]["loss"]:.3f} | {h[0]["train"]["mae"]:.3f}→{h[-1]["train"]["mae"]:.3f} | {h[4]["val"]["mae"]:.3f}→{h[-1]["val"]["mae"]:.3f} | {h[-1]["val"]["mae"]-r["best_val_mae"]:+.4f} |')
for d in DS:
 for z in [7,17,27]:
  fig,axs=plt.subplots(2,2,figsize=(12,8)); specs=[('train','loss'),('train','mae'),('val','mae'),('val','rmse')]
  for n,r in rows.items():
   if r['dataset']!=d or r['seed']!=z:continue
   for ax,(sp,m) in zip(axs.flat,specs):
    pts=[x for x in curves[n] if x.get(sp)];ax.plot([x['epoch'] for x in pts],[x[sp][m] for x in pts],label=r['method'],linewidth=1.3)
  for ax,(sp,m) in zip(axs.flat,specs):ax.set_title(f'{DS[d]} seed{z}: {sp} {m.upper()}');ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.tight_layout();name=f'{d}_seed{z}_curves.png';fig.savefig(ASSETS/name,dpi=160);plt.close(fig);lines += ['',f'![{DS[d]} seed{z} 训练与验证曲线]({ASSETS.name}/{name})']
lines += ['', '观测记录：TaxiBJ K04 三个 best 均在 epoch100，但 seed27 的最终 train MAE=16.928、val MAE=16.596，分别高于同 seed K02 的 train MAE=14.636、val MAE=14.672。不能仅用“已过拟合”解释 seed27 反转。BikeNYC 的 best epoch 集中在 90–100，K02 的 ID 波动小于 K01；验证和测试来自不同时间段、mask，不把二者的绝对差距直接作为过拟合证据。',
 '', '### 2.8 固定尺度、动态专家及门诊断',
 '', '全部六套测试中，配置尺度路径占比均为 1，调用均为 8、面积代理与路径一致。以下动态性指专家选择，不能把专家路径变化说成自由尺度路由。Top-2 的真实路径是四轮专家对路径；只记录 argmax 专家会丢失另一专家的信息。',
 '', '| 实验 | 专家对路径数 | 最集中对路径占比 | 对路径熵（自然对数） | 后三轮门绝对均值 | 后三轮门有符号均值 |','|---|---|---|---|---|---|']
diagnostics=[]
for n,r in rows.items():
 m=evals[n]['sets']['in_distribution']['metrics'];ab=[m[f'coe_step{i}_memory_gate_abs_mean'] for i in [2,3,4]];sg=[m[f'coe_step{i}_memory_gate_signed_mean'] for i in [2,3,4]]
 lines.append(f'| {n} | {int(m["coe_pair_path_unique_count"])} | {m["coe_pair_path_max_fraction"]:.4f} | {m["coe_pair_path_entropy"]:.4f} | '+ '/'.join(f'{x:.4f}' for x in ab)+' | '+ '/'.join(f'{x:+.4f}' for x in sg)+' |')
 for i in range(1,5):
  for exp in ['T','S','TD','SD','TA','ST','TL','SL']:
   diagnostics.append({'name':n,'round':i,'expert':exp,'selection_rate':m[f'coe_step{i}_{exp}_selection_rate'],'weighted_usage':m[f'coe_step{i}_{exp}_usage'],'router_entropy':m[f'coe_step{i}_router_entropy']})
write(ASSETS/'expert_usage.json',diagnostics)
lines += ['', '每轮选择率之和为 2，混合加权 usage 之和为 1；二者不是同一指标。逐轮八专家选择率/加权用量及 Router 熵见 '+asset('expert_usage.json','expert_usage.json')+'。完整逐协议、逐家族、条件化路径及门诊断见 '+asset('full_evaluations.json.gz','完整评估快照（gzip JSON）')+'。训练曲线数据见 '+asset('curves.json','curves.json')+'。',
 '', 'K04 TaxiBJ 后三轮门绝对均值约 0.087–0.099，接近 0.1 边界；K07 的绝对均值约 0.1，但有符号均值约 +0.0469，表示不同通道可取不同符号，不能读成全通道统一 +0.1。接近边界是实测诊断，不直接证明它导致 OOD 退步。',
 '', '### 2.9 逐轮误差变化：不存在必降保证',
 '', '日志 harm/benefit 是相对上一轮预测的缺失点绝对误差增加/减少量，分别对正负部分求全数据集均值；不是“变差点的比例”。direct 且无接受门时 candidate/accepted 相同。净 MAE 变化=harm−benefit，正数表示该轮平均误差增大；不能用隐藏状态变化幅度代替误差改善。',
 '', '| 实验 | 第1轮净 MAE变化 | 第2轮 | 第3轮 | 第4轮 |','|---|---|---|---|---|']
for n in rows:
 m=evals[n]['sets']['in_distribution']['metrics'];net=[m[f'coe_step{i}_candidate_harm']-m[f'coe_step{i}_candidate_benefit'] for i in range(1,5)]
 lines.append(f'| {n} | '+' | '.join(f'{x:+.4f}' for x in net)+' |')
lines += ['', '例如 TaxiBJ K04 seed7 第2轮净变化约 +170.225，后两轮为 −126.881/−180.320；BikeNYC K04 seed27 第3轮净变化约 +4.826。该观测直接排除了“每轮误差必然下降”的表述；本批只监督最终输出。',
 '', '## 3. 配对分析、证据边界与下一步建议',
 '', '### 3.1 预先规定的四组核心配对',
 '', '下表变化定义为 100×(候选/对照−1)，负数为改善。先对每个训练 seed 计算，再报告均值和样本标准差；这不同于“两组三种子均值相除”。绝对差为候选−对照，保留 MAE/RMSE 全部逐 seed。**† 表示退步超过 5%**，包括 mask 复测。',
 '', '| 数据集 | 候选/对照 | seed | 协议 | MAE 绝对差 | MAE 相对变化 | RMSE 绝对差 | RMSE 相对变化 |','|---|---|---|---|---|---|---|---|']
paired=[]
for d in DS:
 for a,b,label in PAIRS:
  for z in [7,17,27]:
   for q in SETS:
    dif={m:metric(d,a,z,q,m)-metric(d,b,z,q,m) for m in ['mae','rmse']};rel={m:change(d,a,b,z,q,m) for m in ['mae','rmse']}
    paired.append({'dataset':d,'a':a,'b':b,'seed':z,'set':q,'delta':dif,'relative_percent':rel})
    lines.append(f'| {DS[d]} | {a}/{b}：{label} | {z} | {SETS[q]} | {dif["mae"]:+.4f} | {pct(rel["mae"])} | {dif["rmse"]:+.4f} | {pct(rel["rmse"])} |')
lines += ['', '| 数据集 | 配对 | 协议 | MAE 相对变化均值 ± SD | MAE 改善种子数 | RMSE 相对变化均值 ± SD |','|---|---|---|---|---|---|']
for d in DS:
 for a,b,label in PAIRS:
  for q in SETS:
   ma=[change(d,a,b,z,q,'mae') for z in [7,17,27]];rm=[change(d,a,b,z,q,'rmse') for z in [7,17,27]]
   lines.append(f'| {DS[d]} | {a}/{b} | {SETS[q]} | {st.mean(ma):+.2f}% ± {st.stdev(ma):.2f}%'+(' **†**' if st.mean(ma)>5 else '')+f' | {sum(x<0 for x in ma)}/3 | {st.mean(rm):+.2f}% ± {st.stdev(rm):.2f}%'+(' **†**' if st.mean(rm)>5 else '')+' |')
write(ASSETS/'paired_metrics.json',paired)
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
for ax,d in zip(axs,DS):
 x=np.arange(4)
 for ix,z in enumerate([7,17,27]):ax.bar(x+(ix-1)*.23,[change(d,a,b,z,'in_distribution','mae') for a,b,_ in PAIRS],width=.22,label=f'seed{z}')
 ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,[a+'/'+b for a,b,_ in PAIRS]);ax.set_title(DS[d]+' ID paired MAE change');ax.set_ylabel('Change (%) : negative is better');ax.grid(axis='y',alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(ASSETS/'paired_id.png',dpi=170);plt.close(fig)
lines += ['',f'![逐训练种子 ID 配对变化]({ASSETS.name}/paired_id.png)',
 '', '### 3.2 固定多尺度的独立贡献',
 '', '**TaxiBJ：CMFF 的尺度收益获得较强的本批支持。** 无差分 K02/K01 的三个 ID 改善为 7.36%/19.67%/10.94%，平均 12.65%；三项主要 OOD 逐 seed 均改善，平均 MAE 降低约 19.45%/31.04%/24.11%，复测方向保留。有差分 K04/K03 的 ID 三 seed 均改善、平均 12.85%；但 seed27 两元/三元分别退步 5.20%/10.71%，因此不能把有差分时的尺度收益说成每个 OOD 条件都更好。',
 '', '**BikeNYC：收益较小且有 mask 敏感性。** 无差分 K02/K01 平均 ID 改善 2.55%，seed17/27 改善 5.15%/2.75%，seed7 略退步 0.26%；三项主要 OOD 平均改善 2.14%/5.74%/2.68%，但两元复测平均反而退步 0.84%。有差分 K04/K03 三个 ID 都改善，平均 3.17%，不能由此推出差分有贡献——这个配对仍然只检验尺度。',
 '', '两个数据集无差分尺度配对都达到事先 ≥1% 的平均实际收益信号，但 BikeNYC 不满足“所有 seed 都改善”。CMFF 可以保留为经过本批验证的固定调度；本批只比较 CMFF/FFFF，不能声称 CMFF 是所有尺度路径的最优解。四轮动态 Top-2 是共同骨干，本批没有单层 Top-8 或固定专家路径对照，不能仅凭本批证明多轮或动态专家选择本身优于对应替代。',
 '', '成本方面，无差分 CMFF 的平均训练＋验证分钟：TaxiBJ 89.10→62.39，BikeNYC 4.64→3.54；平均峰值显存分别 11.105→7.499 GiB、3.195→2.185 GiB。逐 seed 平均时间变化约 −29.75%/−23.63%，显存约 −32.47%/−31.61%。前向实测也较低，但不把面积代理减少 42.19%等同于所有算子的 FLOPs 或延时均降低该比例。',
 '', '### 3.3 受控差分的独立贡献与负结果',
 '', '**TaxiBJ：本批没有复现历史 U 系列的三训练种子一致 ID 收益。** CMFF K04/K02 的 ID 分别变化 −8.48%/−4.25%/+11.61%，平均配对改善只有 0.37%，低于原定 1%；RMSE 平均反而退步约 0.19%。MAE 均值 14.0508→13.9761 接近，但 SD 从 0.6311 增至 1.2811。不能只保留 seed7/17，也不能把均值接近解释成稳定获益。FFFF K03/K01 也出现 seed27 退步 7.48%，平均 MAE 仅改善 0.41%、RMSE 退步约 0.98%。',
 '', '更明确的负结果是 TaxiBJ CMFF 差分的泛化代价：三个 seed 的两元/形态/三元 MAE 全部退步，平均分别 +29.17%/+28.00%/+33.49%，两元/形态复测平均 +30.24%/+26.68%；RMSE 同样显著退步。固定尺度的优势不能用于替代差分的独立证据，K04 比 K01 更好不等于差分比 K02 更好。',
 '', '**BikeNYC：差分未形成 ID 正收益，但组合 OOD 有局部正向权衡。** K04/K02 三个 seed 的 ID 都退步（+1.00%/+1.46%/+1.91%），平均 +1.46%，RMSE 平均 +2.22%；两元/三元 MAE 平均改善 3.50%/6.93%，两元复测改善 5.81%。形态 MAE 平均退步 3.85%，形态复测退步 6.34%，对应 RMSE 平均退步 7.44%/8.63%。FFFF K03/K01 三个 ID 也全部退步，平均 +2.10%。所以不能写成“差分在两个数据集上稳定增强精度和泛化”，也不能说所有 OOD 都无效。',
 '', '历史 U 系列记录过固定 CMFF 差分的三个种子 ID 改善约 5.02%/4.48%/2.64%。本批采用严格冻结、共同初始化和确定性执行协议，其公平内部配对给出不同结果。两批不合并求均值，也不以历史正结果覆盖本批负结果。当前核验未发现本批配置、mask、初始化或 checkpoint 损坏；数值轨迹/训练协议差异只是待检验解释，尚不能归因于某一个因素。',
 '', '潜在机制假设包括门接近边界、状态差分携带对 ID 优化有用但对某些 OOD 不利的方向，以及 direct 链与差分的交互；本批没有针对这些假设的独立干预。四轮中间误差可增大，不能据此提出“误差单调修正”的理论叙事。',
 '', '### 3.4 共享池、条件门和 RMS 的单种子消融',
 '', '| 候选/对照（TaxiBJ seed7） | 问题 | ID MAE变化 | ID RMSE变化 | 两元 MAE变化 | 形态 MAE变化 | 三元 MAE变化 |','|---|---|---|---|---|---|---|']
for a,b,label in ABLATIONS:
 vals=[change('taxibj',a,b,7,'in_distribution','mae'),change('taxibj',a,b,7,'in_distribution','rmse')]+[change('taxibj',a,b,7,q,'mae') for q in ['unseen_combinations','unseen_geometry','unseen_triple']]
 lines.append(f'| {a}/{b} | {label} | '+' | '.join(pct(x) for x in vals)+' |')
lines += ['', '单种子六协议完整配对：', '', '| 候选/对照 | 协议 | MAE变化 | RMSE变化 |','|---|---|---|---|']
for a,b,_ in ABLATIONS:
 for q in SETS:lines.append(f'| {a}/{b} | {SETS[q]} | {pct(change("taxibj",a,b,7,q,"mae"))} | {pct(change("taxibj",a,b,7,q,"rmse"))} |')
lines += ['', '**共享池：有明确参数节省，没有本批全面精度优势。** 独立池 K05/K06 的总参数 821175，对应共享池 330807，共享少约 59.72%。无差分独立池 K05 相比 K02 的 ID 改善 8.46%，主要 OOD MAE 略到明显退步（1.51%/6.09%/6.34%）；有差分 K06 相比 K04 的 ID 改善 6.08%，两元 MAE 退步 11.02%、形态改善 4.02%、三元退步 1.84%，部分 RMSE 退步更明显。激活专家次数和面积相同，前向约 5.05/5.07 ms（K02/K05）、5.38/5.47 ms（K04/K06），不能因为参数少就声称前向成本按 59.72%同比降低。',
 '', '**差分代价并非仅在共享池出现。** K06/K05 的 ID 改善 6.10%，两元/形态/三元 MAE 却退步 39.04%/12.02%/30.26%。这与共享池的 ID/OOD 权衡相容，仍只有一个训练 seed，不能证明普遍原因。',
 '', '**状态条件化：必要性未获支持。** K07/K04 的 ID MAE 略改善 0.66%，RMSE近乎相同；三项主要 OOD MAE 改善 3.60%/9.70%/3.80%，复测也同方向。报告中不能把学习输入状态的门视为已证实贡献；其与常数通道门的有效参数和训练轨迹也不同。',
 '', '**RMS：有单 seed 正向证据，但不足以挽救整体差分主张。** K08/K04 取消 RMS 后 ID MAE 高 2.31%，三项 OOD 高 9.66%/2.50%/9.39%，复测同方向。可写成此次 seed7 的幅度匹配消融支持信号，尚无跨 seed/数据集稳定性证明，更不代表 K04 优于无差分 K02。',
 '', '### 3.5 论文主线目前能支持到哪里',
 '', '| 主张 | 本批支持程度 | 可采用的准确表述 |','|---|---|---|',
 '| 固定 CMFF 相比 FFFF 的价值 | 两数据集平均 ID 信号通过，TaxiBJ 更强；成本有实测下降 | 固定多尺度调度在此协议下改善平均精度和成本；承认 BikeNYC seed7 与复测例外 |',
 '| 四轮共享异构专家 CoE | 共享参数节省已测；共同骨干没有独立轮数/路由对照 | 提出并分析共享四轮动态 Top-2 架构，不凭本批声称多轮/动态性各自已被证明必要 |',
 '| ±0.1 受控差分稳定提升 ID | 未通过本批预先标准 | 明确报告跨 seed 波动与跨数据集 ID 退步，暂不写成已确立的稳定核心收益 |',
 '| 差分提升未见缺失泛化 | TaxiBJ 三主要 OOD 退步；BikeNYC 有组合/形态权衡 | 协议依赖的精度与泛化取舍 |',
 '| 条件门优于无条件门 | 单 seed 不支持 | 状态条件化必要性待验证 |',
 '| RMS 匹配必要 | 仅单 seed 正向消融 | 保留为待复验的幅度控制设计 |',
 '| 每轮误差必然下降 | 被本批净误差记录否定 | 多轮表示变换，最终补全监督；不作单调误差保证 |',
 '| 第三个核心贡献已经确定 | 本批未研究/确定 | 保持开放，不将 W 第二视图/一致性或新增 FFN 自动升级为核心贡献 |',
 '', '局限：只有两个数据集、一个训练缺失率 0.4、三个训练 seed；K05–K08 单 seed。多个 best 落在 epoch100，100 epoch 完成不等于已经充分收敛，也不能预言延长训练会消除反转。测试窗口时间相关，mask 复测不是独立训练复验，不把三 seed 的样本 SD 当作统计显著性证明。未做同协议外部 baseline、单层 Top-8、固定专家选择、其他尺度路径、匹配容量/计算的完整消融，不能声称 SOTA、CMFF 全局最优或全部架构部件具备独立创新必要性。当前 OOD 测试已被分析，后续候选若依据这些测试改动，需预先冻结新的验证规则及独立确认，避免在同一测试集上反复选择后称作新证据。',
 '', '### 3.6 下一步建议：先解决前两项证据缺口',
 '', '1. **保留本批完整结果与 CMFF 参考，不改动原队列。** 若后续研究启动，以 K02（CMFF 无差分）和 K04（原受控差分）作为明确参考；不要把新增机制的总收益直接算作差分收益。',
 '2. **优先诊断差分波动，而非立即扩大模型。** 先离线对已有验证日志/检查点分析门强度、输入修正及专家选择与 seed27 反转的关系；只称相关诊断。若需因果验证，先冻结小规模 K02/K04 与常数通道门、较小固定 bound（如 0.05）对照，保持 mask、初始化和预算一致，以验证信息条件化和强度是否分别必要，不从同一 OOD 测试挑 bound。',
 '3. **补足共享结构主张的独立证据。** K02/K05、K04/K06 补独立训练 seed；若要主张多轮或动态专家选择优越，再分别增加同激活预算的单层 Top-8 与训练时固定专家路径对照，并报告总/可训参数、面积代理和实测时间，不能用一次测试强制路径替代从头训练因果对照。',
 '4. **若转向 ID/OOD 综合目标，先在验证时间段定义目标。** 冻结基础/组合/形态验证协议、权重和候选，再独立测试；不能因为 BikeNYC 差分有组合收益就事后更换本批 ID 门槛。',
 '5. **每轮后 64→256→64 FFN 可作为单独结构探索，不是本批结论。** 动机可明确为专家混合后统一通道重整，但专家内部已有升维非线性、TA 内已有 FFN。应比较 K02/K04 各自加同一 FFN、记录新增容量与成本，再决定是否有独立价值；不能将标准 FFN 本身视为足够的第三项创新，也不依据当前结果宣称它能修复差分的 OOD 问题。',
 '', '以上均是建议，本次未启动、停止、恢复或重跑训练/评估，也未更改冻结计划、源码或现有实验输出。']

# Generate an OOD plot with individual training seeds retained.
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
qs=['unseen_combinations','unseen_geometry','unseen_triple'];labels=['Pairs','Geometry','Triple']
for ax,d in zip(axs,DS):
 x=np.arange(3)
 for ix,z in enumerate([7,17,27]):ax.bar(x+(ix-1)*.23,[change(d,'K04','K02',z,q,'mae') for q in qs],width=.22,label=f'seed{z}')
 ax.set_xticks(x,labels);ax.axhline(0,color='black',linewidth=.8);ax.set_title(DS[d]+' CMFF delta: K04/K02 OOD');ax.set_ylabel('MAE change (%) : negative is better');ax.legend();ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(ASSETS/'delta_ood.png',dpi=170);plt.close(fig)
idx=lines.index('### 3.4 共享池、条件门和 RMS 的单种子消融')
lines[idx:idx]=[f'![差分的逐训练种子 OOD 变化]({ASSETS.name}/delta_ood.png)','']
DEST.write_text('\n'.join(lines)+'\n')
print('REPORT',DEST);print('ASSETS',ASSETS);print('TABLE_COUNTS',len(rows),'runs',sum(len(r['test']) for r in rows.values()),'sets',len(families),'family rows',len(paired),'core pair rows');print('LINES',len(lines))
