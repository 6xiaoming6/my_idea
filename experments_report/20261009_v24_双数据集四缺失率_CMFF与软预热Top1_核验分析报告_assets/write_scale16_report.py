import json,shutil,os,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ['MPLCONFIGDIR']='/tmp/v24_matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/home/students/HuangMingYu/code/py/my_idea/my_idea')
S=R/'outputs/v24-COE/experiments/scale_rate_compare/a8e91b58131e845d'
D=json.load(open('/tmp/scale16_data.json'));A=json.load(open('/tmp/scale16_audit.json'))
report=R/'experments_report/20261009_v24_双数据集四缺失率_CMFF与软预热Top1_核验分析报告.md'
assets=report.with_name(report.stem+'_assets');assets.mkdir(exist_ok=True)
for src,name in [('/tmp/scale16_audit.json','verification.json'),('/tmp/audit_scale16.py','audit_scale16.py')]:shutil.copyfile(src,assets/name)
(assets/'paired_metrics.json').write_text(json.dumps(A['pairs'],ensure_ascii=False,indent=2))
auto=R/'experments_report/20261009_v24_双数据集四缺失率尺度对比_a8e91b58131e845d_完整分析.md'
text=auto.read_text().split('## 3. 配对分析、证据边界与建议')[0]
text=text.replace(text.splitlines()[0],'# 20261009 v24：TaxiBJ/BikeNYC四缺失率，CMFF与三尺度软预热Top-1核验分析报告',1)
text=text.replace('完整批次。','完整批次。本文重新核对冻结配置、源码、原始日志和32份检查点，并补充机制与成本分析。',1)
L=[text,'### 2.1 原始记录完整性与预算核验','']
start=datetime.fromtimestamp(A['started']['time'],timezone(timedelta(hours=8)));end=datetime.fromtimestamp(A['completed']['time'],timezone(timedelta(hours=8)))
L += [f'- 16/16组均训练100epoch，共1600epoch；每组20次验证，96/96套测试完成；失败回执0。',
 f'- 32份best/last均可加载，配置哈希、epoch及所有浮点模型张量有限性检查通过；best与原始ID验证最小MAE所在epoch一致。',
 f'- 425份冻结源码/配置哈希一致，当前源码与快照无差异；6份数据文件SHA256与冻结指纹一致。每个配对除尺度策略、软预热及实验元数据外配置一致。',
 '- 主损失/MAE/RMSE全部有限。三个早期诊断出现inf：TaxiBJ 0.6 Top1 epoch2的S专家梯度、TaxiBJ 0.8 CMFF epoch3的S专家梯度、TaxiBJ 0.8 Top1 epoch3的第4轮Router梯度。均伴随AMP跳步，不能把诊断inf隐去，也没有证据将0.6退步单独归因于它。',
 '- TaxiBJ每组AMP跳步11–12/7700次，BikeNYC每组4–5/1600次；0.6 Top1为11次，未表现为异常多的跳步。',
 '- 训练记录证实：Top1 epoch1–10每窗口24次专家执行，epoch11–100为8次；所有验证/测试为8次。CMFF全程8次。预热额外执行量已真实计入，不能称为等训练计算预算。',
 '- 同seed初始化状态哈希相同，总参数330807；CMFF可训练271163、Top1可训练330807。差59644来自可学习尺度头。固定组拥有但冻结该头，因此不是严格等可训练参数比较。',
 f'- 队列墙钟：{start:%Y-%m-%d %H:%M:%S}至{end:%Y-%m-%d %H:%M:%S}（北京时间），约{A["wall_hours"]:.2f}小时。训练+验证计时合计9.62小时，其余包含测试、保存、启动和报告。',
 '- TaxiBJ train/val/test=2452/342/707窗口，77/11/23batch；BikeNYC=511/73/147窗口，16/3/5batch。BikeNYC每组约3.2–4.0分钟与其数据规模、网格和实测计时相符，不是只跑了部分epoch。',
 f'- [逐组检查点/源文件/数据审计]({assets.name}/verification.json)。',
 '', '### 2.2 八个配对的ID与OOD相对变化', '',
 '下表均为Top1相对同数据集、同缺失率CMFF的误差变化，负值为改善。A4为ID/两元/形态/三元MAE的算术平均，只作同一数据集同一率内的辅助汇总；没有用于选择best，也不跨数据集合并原始MAE。', '',
 '| 数据集 | 率 | ID MAE | ID RMSE | 两元MAE | 形态MAE | 三元MAE | A4 CMFF→Top1 | A4变化 |',
 '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
keys=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']
for n,p in A['pairs'].items():
 j=D[n]['job'];d=p['mae_change_pct'];L.append(f'| {j["dataset"]} | {j["rate"]} | {d[keys[0]]:+.2f}% | {p["rmse_change_pct"]:+.2f}% | {d[keys[1]]:+.2f}% | {d[keys[2]]:+.2f}% | {d[keys[3]]:+.2f}% | {p["A4_cmff_top1"][0]:.3f}→{p["A4_cmff_top1"][1]:.3f} | {p["A4_change_pct"]:+.2f}% |')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,ds in zip(axs,['taxibj','bikenyc']):
 for method,label in [('cmff','Fixed CMFF'),('top1','Soft-start Top1')]:
  ys=[D[f'{ds}_r{r:02}_{method}']['evaluation']['sets']['in_distribution']['metrics']['mae'] for r in [20,40,60,80]]
  ax.plot([.2,.4,.6,.8],ys,'o-',label=label)
 ax.set_title(ds);ax.set_xlabel('Train / ID-test missing rate');ax.set_ylabel('ID MAE');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(assets/'id_by_rate.png',dpi=150);plt.close(fig)
L += ['',f'![ID across rates]({assets.name}/id_by_rate.png)','', '### 2.3 逐轮尺度概率、路径与成本', '',
 '以下F/M/C表示各数据集的原网格、1/2、1/4。TaxiBJ与BikeNYC绝对尺寸不同。ID路径为best检查点的707/147个窗口统计；组内专家权重仍可随输入变化，固定尺度路径不等于整个模型输入无关。', '',
 '| 自由组 | ID全部非零尺度路径 | ID面积代理 | 相对4.625 | ID前向ms/窗 CMFF→Top1 | 训练分钟 CMFF→Top1 |',
 '| --- | --- | --- | --- | --- | --- |']
for n,d in D.items():
 if d['job']['method']!='top1':continue
 b=D[n.replace('_top1','_cmff')];m=d['evaluation']['sets']['in_distribution']['metrics'];bm=b['evaluation']['sets']['in_distribution']['metrics']
 paths={k[len('coe_scale_path_'):-len('_fraction')].upper():v for k,v in m.items() if re.fullmatch(r'coe_scale_path_[fmc]{4}_fraction',k) and v>0}
 ps='；'.join(f'{k} {v*100:.2f}%' for k,v in paths.items())
 L.append(f'| {n} | {ps} | {m["coe_expert_grid_equivalents"]:.3f} | {(m["coe_expert_grid_equivalents"]/4.625-1)*100:+.2f}% | {bm["forward_ms_per_sample_per_rank"]:.3f}→{m["forward_ms_per_sample_per_rank"]:.3f} | {b["receipt"]["total_time_sec"]/60:.1f}→{d["receipt"]["total_time_sec"]/60:.1f} |')
L += ['', 'TaxiBJ 0.2/0.4的CFFF、TaxiBJ 0.8的CCCF、BikeNYC 0.6的CMCF在六套测试中均占100%。其余四组ID仅2–3条非零路径。没有模型在ID使用超过3条路径；这不是要求必须遍历81条，而是说明收益不能直接归因于丰富的条件化路径。',
 'TaxiBJ自由组峰值约15.67–15.73GiB，对照约8.04–8.05GiB；BikeNYC约4.52–4.54GiB，对照约2.33–2.36GiB。峰值主要包含三尺度软预热，不能当作Top1推理显存。前向延迟来自评估过程的计时，未作独立重复性能基准，不根据几个百分点差值作强效率结论。', '',
 '| 自由组 | 第1轮概率F/M/C | 第2轮 | 第3轮 | 第4轮 |', '| --- | --- | --- | --- | --- |']
for n,d in D.items():
 if d['job']['method']!='top1':continue
 m=d['evaluation']['sets']['in_distribution']['metrics'];prob=['/'.join(f'{m[f"coe_step{i}_{s}_probability"]:.4f}' for s in ('fine','mid','coarse')) for i in range(1,5)]
 L.append('| '+n+' | '+' | '.join(prob)+' |')
L += ['', '### 2.4 全部训练趋势与预热切换', '',
 '| 组 | val@5 | val@10 | val@20 | val@40 | val@60 | val@80 | val@100 | best epoch | 最后train MAE | AMP跳步 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for n,d in D.items():
 h=d['history'];vals=[f'{h[e-1]["val"]["mae"]:.3f}' for e in [5,10,20,40,60,80,100]]
 L.append('| '+n+' | '+' | '.join(vals)+f' | {d["receipt"]["best_epoch"]} | {h[-1]["train"]["mae"]:.3f} | {sum(x["train"].get("train_skipped_amp_steps",0) for x in h):.0f} |')
L += ['', 'TaxiBJ 0.6 Top1在epoch10 val=47.549，epoch15=50.458，之后缓慢降至best epoch90的42.537；同时epoch100 train MAE仍45.662。明显退步同时存在于训练和验证，不能解释为单纯测试mask偶然性或只发生过拟合。',
 '该组第三轮在epoch5验证时已100%选C；第四轮选择C的比例从epoch10的16.37%升至epoch15的75.73%，epoch40后100%。它在预热转稀疏后逐渐锁定后两轮C，最终ID为FFCC/CFCC。只能说这一现象与较低最终空间分辨率相容，不能仅凭相关性断言全部性能损失由最后一轮C造成。',
 'BikeNYC 0.6在epoch10之后验证尺度路径最大占比持续为100%；其ID优势来自单一路径模型，不是后期不断适应不同样本。多数组best在90–100epoch，没有证据支持单纯多训几轮能解决所有失败。', '',
 '### 2.5 ID逐缺失族MAE', '',
 '| 组 | 随机点 | 节点中断 | 时间缺口 | 空间区域 |', '| --- | --- | --- | --- | --- |']
for n,d in D.items():
 m=d['evaluation']['sets']['in_distribution']['metrics'];L.append('| '+n+' | '+' | '.join(f'{m["coe_family_"+f+"_mae"]:.4f}' for f in ['random_point','node_outage','temporal_gap','spatial_region'])+' |')
L += ['', 'TaxiBJ 0.2的总体改善主要来自节点和区域缺失，随机点和时间缺口反而略退步。BikeNYC 0.6随机点/时间/区域改善，节点中断略退步。TaxiBJ 0.6的四个族均明显变差。总体MAE按有效缺失位置统计，不能用四族未经计数加权的均值替代。', '',
 '### 2.6 六套评估逐缺失族MAE/RMSE完整附表', '', '| 组 | 协议 | 缺失族 | MAE | RMSE |', '| --- | --- | --- | --- | --- |']
for n,d in D.items():
 for key,e in d['evaluation']['sets'].items():
  m=e['metrics']
  for k,v in m.items():
   if k.startswith('coe_family_') and k.endswith('_mae'):
    family=k[len('coe_family_'):-4];rmse=m.get('coe_family_'+family+'_rmse')
    L.append(f'| {n} | {key} | {family} | {v:.4f} | {rmse:.4f} |')
L += ['', '## 3. 分析、证据边界与下一步建议', '',
 '### 3.1 这次是否推翻“CMFF更可靠”的判断', '',
 '**没有。扩大到两个数据集和四个训练缺失率后，固定CMFF仍是更可靠的默认骨干。自由Top1有两个值得复验的ID正信号，但没有形成普遍优势。**',
 '八个配对中，CMFF在5个率上ID MAE更低；Top1在TaxiBJ0.2和BikeNYC0.6分别改善4.25%和3.32%；TaxiBJ0.4只低0.002894（0.02%），应视为数值接近，不能当作实质性胜利。RMSE方向与这些总体判断一致。',
 '若看ID/两元/形态/三元四项MAE平均，Top1在TaxiBJ0.2、TaxiBJ0.8、BikeNYC0.6三个配对更好，另外五个配对更差。这个汇总揭示权衡，不应覆盖逐协议退步，也不能跨数据集平均原始误差。', '',
 '### 3.2 三个有价值的信号分别意味着什么', '',
 '**TaxiBJ0.2：低缺失率下CFFF是可研究的路径候选。** ID MAE12.250→11.730，RMSE21.050→20.098；两元/三元MAE降低5.11%/17.18%，形态MAE略升0.91%。形态复测则降低1.02%，说明微小差异对mask敏感。原两元RMSE反而升约5.9%，不能说所有误差指标都改善。CFFF相较CMFF多一轮F，面积代理增加32.43%、训练约56.1→86.2分钟；收益可能涉及更高分辨率计算、软预热训练或具体路径，尚未隔离路由本身。',
 '**BikeNYC0.6：本轮最干净的局部正结果。** ID MAE3.8313→3.7042、RMSE14.6641→14.2726，两元/形态/三元MAE分别改善7.19%/8.18%/11.03%，两套mask复测均同方向；面积代理降低40.54%。不过六套测试全部固定CMCF。它首先支持“CMCF可能更适合该条件”，并不能证明每窗口动态尺度选择的价值。',
 '**TaxiBJ0.8：效率和泛化权衡，而非更高ID精度。** Top1锁定CCCF，面积代理约减48.65%、实测前向约减22.31%；三项OOD MAE改善14.13%–22.43%，但ID退步10.67%，四个ID缺失族均变差。若应用要求ID优先，不能将它替代CMFF；若重视推理预算或分布迁移，可将固定CCCF作为下一步匹配参考。软预热使总训练仍约48.3→72.5分钟，不是全面更省计算。', '',
 '### 3.3 两个需要明确保留的负结果', '',
 '**TaxiBJ0.4：ID几乎不变，组合泛化明显变差。** 两元MAE15.519→30.745（+98.12%）、形态20.043→31.377（+56.55%）、三元15.304→31.500（+105.83%）。两套新mask复测仍退步92.96%/57.81%，不是单一测试mask现象。验证MAE也略高1.51%，没有证据支持在当前0.4主任务上更换骨干。',
 '**TaxiBJ0.6：当前自由训练方案明显失败，但训练流程没有中断。** ID MAE15.229→39.839（+161.60%），六套协议都大幅退步，后两轮选择C且训练误差也高。冻结源码显示最终解码为逐点LayerNorm与1×1卷积，末轮C后只上采样再解码，没有额外细网格空间修复阶段；这提供了一个值得验证的空间分辨率瓶颈解释。仍需匹配固定FFCC/CFCC或受约束路径重新训练才能检验，不能用测试时强制改路径的结果代替结构因果对照。', '',
 '### 3.4 对软预热与条件化选择能说到哪一步', '',
 '记录已证实预热确实执行了三个尺度、尺度头有训练，之后实际转为单尺度；因此这次负结果不能解释成“根本没有执行软预热”。但本批没有同条件无预热自由Top1对照，不能判断5+5软预热相对冷启动本身带来了多少收益或损失。',
 '四组在全部六协议都退化为单一尺度路径，另四组也仅少数路径；有路径变化的TaxiBJ0.6/BikeNYC0.2/0.4/0.8均未获得ID优势。两组有明确ID收益的模型反而都是单一路径。这与“自由路由目前更像全局路径选择器”的解释一致，但不等于证明所有自适应尺度方法无效。',
 '不能因训练中使用了不同尺度、或概率随输入变化，就声称条件化路径带来收益；需要与训练好的固定路径模型对照。也不能为了证明自由性而强制均衡或追求81条路径覆盖。', '',
 '### 3.5 公平性与外推范围', '',
 '- 单seed7，每数据集每个率独立重训；同mask复测不增加训练种子数，也不是跨数据集零样本迁移。',
 '- 两组同总参数与共同初始化，但可训练参数不同，Top1前10epoch额外执行三尺度；相同100epoch不等于相同训练FLOPs。',
 '- 两个数据集分别只有707/147个测试窗口；时间窗口可能相关，不将它们当作独立重复声称显著性。',
 '- 原四基础族训练，未加入W19–W22第二视图。本批比较不能与旧W20/W22不同训练暴露的数字混作同预算结论。',
 '- 新CMFF0.4结果与旧W01并不完全相同；本批执行模块不同，仅用本批配对参考解释改进，不将严格确定性误解为跨实现数学等价就一定得到同一训练轨迹。',
 '- 每个缺失率均更换了训练分布并重训，OOD误差不必随率单调；跨率非单调不自动证明mask生成错误。', '',
 '### 3.6 下一步：收缩到能回答原因的对照', '',
 '1. **默认主干继续CMFF＋±0.1差分。** 当前没有证据支持用自由Top1普遍替换它，尤其TaxiBJ0.4和0.6。',
 '2. **优先验证BikeNYC0.6固定CMCF。** 用相同初始化从头训练固定CMCF，与CMFF及本次自由模型比较；同时补一个保持相同前10epoch软阶段、随后固定CMCF的对照。两者分别区分最终路径与软预热优化过程，不能只换已训练模型推理路径。',
 '3. **TaxiBJ0.2固定CFFF作第二候选。** 控制多一轮F带来的面积与实际运行成本，再比较固定CFFF与自由模型；对两个正信号补seed17/27，先用验证选择而非反复使用测试挑方法。',
 '4. **TaxiBJ0.8 CCCF作为精度/成本分支。** 只有在明确接受ID损失的应用目标下继续，不将OOD收益包装成全面精度提升。',
 '5. **不再立即扩展更多无约束自由路由。** 若固定发现路径已能达到自由模型效果，就采用更简单、可解释的固定计算计划，把主线继续放在受控跨轮通信和已经出现正信号的观测变化训练。',
 '', '本次仅核验、分析和导出报告，没有启动、停止或重跑实验。']
report.write_text('\n'.join(L)+'\n')
shutil.copyfile('/tmp/write_scale16_report.py',assets/'write_scale16_report.py')
print(report)
print('lines',len(report.read_text().splitlines()))
