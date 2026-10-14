#!/usr/bin/env python3
"""All nine runs + frozen B1/B2 references, methods before evidence and analysis."""
from __future__ import annotations
import csv,json,math,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
import torch
from run_b3_c3 import load,write,digest


def fmt(x):return f'{x:.4f}' if isinstance(x,(int,float)) else '—'


def collect(suite,n,j):
    rp=suite/'results'/f'{n}.json';ep=suite/'evaluations'/f'{n}.json';pointer=suite/'runs'/f'{n}.json'
    r=load(rp) if rp.exists() else {};e=load(ep) if ep.exists() else {}
    run=Path(r['run_dir']) if r else Path(load(pointer)['run_dir']) if pointer.exists() else None
    history=[];meta={};issues=[];cpinfo={}
    if run:
        hp=run/'logs/metrics.jsonl'
        if hp.exists():history=[json.loads(v) for v in hp.read_text().splitlines() if v.strip()]
        if (run/'training_metadata.json').exists():meta=load(run/'training_metadata.json')
        for label in ('best','last'):
            cp=run/'checkpoints'/f'{label}.pth'
            if cp.exists():
                saved=torch.load(cp,map_location='cpu',weights_only=False)
                cpinfo[label]={'epoch':saved['epoch'],'config_match':digest(saved['config'])==digest(j['config']),
                    'finite':all(bool(torch.isfinite(t).all()) for t in saved['model'].values() if torch.is_floating_point(t))}
                del saved
        if any(not v['config_match'] or not v['finite'] for v in cpinfo.values()):issues.append(n+': checkpoint config/nonfinite error')
        if r and (set(cpinfo)!={'best','last'} or cpinfo['last']['epoch']!=j['config']['train']['epochs'] or cpinfo['best']['epoch']!=r['best_epoch']):issues.append(n+': checkpoint/receipt mismatch')
    expected=j['config']['train']['epochs'];protocol=j['config']['experiment_plan']['protocol']
    complete=r.get('status')=='finished' and e.get('status')=='finished' and set(e.get('sets',{}))==set(protocol['evaluations'])
    if r and r.get('config_sha256')!=digest(j['config']):issues.append(n+': receipt config mismatch')
    if e and (e.get('config_sha256')!=digest(j['config']) or e.get('protocol_sha256')!=digest(protocol)):issues.append(n+': evaluation config/protocol mismatch')
    if complete and (e.get('best_epoch')!=r['best_epoch'] or e.get('checkpoint')!=str((run/'checkpoints/best.pth').resolve()) or e.get('npz')!=str(Path(j['sources']['test']).resolve())):issues.append(n+': evaluation source/checkpoint mismatch')
    if complete and [h['epoch'] for h in history]!=list(range(1,expected+1)):issues.append(n+': missing/duplicate epochs')
    if r and r.get('completed_epochs')!=expected:issues.append(n+': training budget mismatch')
    for h in history:
        for split in ('train','val'):
            for key in ('mae','rmse','loss'):
                v=(h.get(split) or {}).get(key)
                if v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v)):issues.append(f'{n}: nonfinite {split}.{key} epoch{h["epoch"]}')
    for label,ev in e.get('sets',{}).items():
        if any(not isinstance(ev['metrics'].get(k),(int,float)) or not math.isfinite(ev['metrics'][k]) for k in ('mae','rmse')):issues.append(n+': nonfinite test '+label)
    state='完成' if complete and not issues else '证据异常' if issues else '训练完成，评估未齐' if r else '部分训练' if history else '未开始'
    failure=load(suite/'failures'/f'{n}.json') if (suite/'failures'/f'{n}.json').exists() else None
    row={'name':n,'method':j['method'],'state':state,'complete':complete and not issues,'epochs':history[-1]['epoch'] if history else 0,
        'run':str(run) if run else None,'best_epoch':r.get('best_epoch',cpinfo.get('best',{}).get('epoch')),
        'val_mae':r.get('best_val_mae'),'params':meta.get('total_params'),'trainable':meta.get('trainable_params'),
        'minutes':r.get('total_time_sec',sum(h.get('perf',{}).get('epoch_time_sec',0) for h in history))/60,
        'peak_gib':max((h.get('perf',{}).get('peak_memory_gb',0) for h in history),default=0),
        'failure':failure,'checkpoint_audit':cpinfo}
    return row,history,e,issues


def export_report(root,suite,js):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite);records=[];curves={};evaluations={};issues=[];results=[];families=[]
    protocol=next(iter(js.values()))['config']['experiment_plan']['protocol']
    for n,j in js.items():
        if j['config']['experiment_plan']['protocol']!=protocol:raise ValueError('Inconsistent suite protocols')
        row,h,e,bad=collect(suite,n,j);records.append(row);curves[j['method']]=h;evaluations[j['method']]=e;issues+=bad
    references={}
    for method in ('B1','B2'):
        directory=suite/'references'/method
        if not (directory/'evaluation.json').exists():issues.append('missing reference '+method);continue
        e=load(directory/'evaluation.json');r=load(directory/'receipt.json');c=load(directory/'config.json')
        if e.get('status')!='finished' or e.get('protocol_sha256')!=digest(protocol) or set(e.get('sets',{}))!=set(protocol['evaluations']) or e.get('config_sha256')!=digest(c) or r.get('config_sha256')!=digest(c):issues.append('invalid frozen reference '+method)
        references[method]=r;evaluations[method]=e
        curves[method]=[json.loads(v) for v in (directory/'history.json').read_text().splitlines() if v.strip()]
    for method,e in evaluations.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];results.append({'method':method,'protocol':label,'mae':m['mae'],'rmse':m['rmse'],'samples':ev['samples'],
                'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank'),
                'expert_calls':m.get('coe_expert_execution_count'),'area':m.get('coe_expert_grid_equivalents')})
            for key,v in m.items():
                if key.startswith('coe_family_') and key.endswith('_mae'):
                    prefix=key[:-3];families.append({'method':method,'protocol':label,'family':key[len('coe_family_'):-4],
                        'mae':v,'rmse':m.get(prefix+'rmse'),'count':m.get(prefix+'count')})
    final=len(js)==9 and len(records)==9 and all(r['complete'] for r in records) and not issues
    phase='完整' if final else '阶段';date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    stem=date+'_v25_尺度选择专家数量与差分语义九组探索_'+suite.name
    dest=root/'experments_report';dest.mkdir(parents=True,exist_ok=True);assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for filename,value in [('summary.json',records),('curves.json',curves),('evaluations.json',evaluations),('audit.json',{'complete':final,'issues':issues})]:write(assets/filename,value)
    for filename,rows in [('protocols.csv',results),('families.csv',families)]:
        if rows:
            with (assets/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for method,h in curves.items():
        if not h:continue
        axes[0].plot([v['epoch'] for v in h],[v['train']['mae'] for v in h],label=method)
        vs=[v for v in h if v.get('val')];axes[1].plot([v['epoch'] for v in vs],[v['val']['mae'] for v in vs],label=method)
    for ax,title in zip(axes,('Training MAE','ID validation MAE')):
        ax.set(title=title,xlabel='Epoch',ylabel='MAE');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=160);plt.close(fig)
    cfg=next(iter(js.values()))['config']
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}；训练 seed7；新增预算9组。队列：`{suite}`。','',
        '## 1. 实验具体做法与复现信息','',
        '|组|对照|尺度与 Top-k|差分输入轮次|具体做法|','|---|---|---|---|---|']
    for j in js.values():
        s=j['config']['model']['coe']['scale_delta_exploration']
        lines.append(f'|{j["method"]}|{j["reference"]}|{s["path"]}，{s["top_k_per_step"]}|{s["delta_steps"]}|{j["config"]["experiment_plan"]["method"]}|')
    lines+=['','B1/B2 是上一批已完成参考，不是本批新增训练。S04 从相同随机初始化独立训练，未加载旧 B2 已训练参数。','',
        '### 共同数据、模型与训练配置','',
        '真实 F/M/C 输入分别为 [B,2,12,32,32]、[B,2,12,16,16]、[B,2,12,8,8]；同一时间 split 内完整 F 预构造 M/C 空间求和，然后各尺度独立生成观测掩码。输入编码时按已知面积4/16转换数值单位，无数据拟合归一化。完整 F 只用于监督，不进入模型前向；隐藏占位值被掩码过滤。',
        f'训练四族：{cfg["data"]["train_mask_diversity"]["families"]}，各尺度缺失率0.4；训练逐epoch重采样、验证/测试固定。沿用上一批数据及mask种子；已见单一结构属于ID。完整测试协议：{list(protocol["evaluations"])}。',
        'H0 融合全部尺度编码；自由尺度选择决定本轮计算和观测重新读取，不意味着首次看到某尺度。四轮共享八种异构专家 T/S/TD/SD/TA/ST/TL/SL。选中专家按集合内 softmax 融合；Top-8是八专家加权融合。各轮输出恢复到F网格。无额外跨轮残差、历史池、FFN、乘性交互、第二视图、一致性、粗尺度损失或中间轮监督；数值后备K0固定。',
        'S01 每轮每样本只执行一个硬选尺度，无软尺度预热，选中分支使用有偏 straight-through 近似梯度。S03 首轮Top-1用选中概率ST系数提供任务路由梯度，前向系数仍为1；这是需要披露的训练估计器差异。',
        '差分为当前与上一轮输入隐藏状态之差，RMS幅度匹配比例detach，±0.1共享条件门末层零初始化，仅修正专家输入。CMFF第四轮差分H3−H2来自F与M输出；CFFF第四轮该差分来自两个F输出。所有单位置组只改变启用位置，历史更新不变。',
        f'{cfg["train"]["epochs"]} epoch，batch{cfg["data"]["batch_size"]}，每{cfg["train"]["val_epoch"]}epoch验证，seed7，AdamW，lr1e-3余弦到3e-4，weight_decay1e-4，clip1，AMP，严格确定性；只按ID验证MAE选best，last保存完整恢复状态。损失为F缺失位置L1+0.01 candidate均衡。新增训练和评估物理GPU1单卡串行；不要求GPU独占，成本不能视为严格独占速度对照。',
        'S02每窗口16次专家调用，S03为15次，其细尺度参与更多；不能声称二者等计算。无差分共享Top-2参考每窗口8次。总参数与可训参数分别报告。','',
        '### 来源与复现','',f'冻结计划：`{suite/"plan.json"}`；源码：`{suite/"source_snapshot"}`；初始化：`{suite/"initialization_audit.json"}`；引用证据：`{suite/"references"}`。',
        f'恢复命令：`python -u {suite/"source_snapshot/scripts/v25/run_scale_delta_exploration.py"} --suite {suite} --gpu 1`。完成训练跳过，缺评估补齐。','',
        '|组|配置|运行目录|','|---|---|---|']
    lines += [f'|{r["method"]}|`{suite/"configs"/(r["name"]+".json")}`|`{r["run"]}`|' for r in records]
    for method,r in references.items():lines.append(f'|旧{method}|`{suite/"references"/method/"config.json"}`|`{r["run_dir"]}`|')
    lines += ['','## 2. 实验结果','','|组|状态|epoch|best epoch|验证MAE|分钟|总参数/可训|峰值GiB|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["state"]}|{r["epochs"]}|{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in records]
    lines += ['','|组|测试协议|MAE|RMSE|ms/窗口|专家次数|面积代理|mask seed|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{fmt(r["expert_calls"])}|{fmt(r["area"])}|{r["mask_seed"]}|' for r in results]
    lines += ['','### ID/OOD逐缺失族结果','','|组|协议|缺失族|MAE|RMSE|缺失计数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    lines += ['','### 训练曲线与机制诊断','',f'![训练和验证曲线]({assets.name}/curves.png)','',
        f'逐epoch记录：[curves.json]({assets.name}/curves.json)；所有测试、路由路径、门、RMS匹配与观测残差：[evaluations.json]({assets.name}/evaluations.json)。','',
        '|组|best验证MAE|末次验证MAE|','|---|---|---|']
    for method,h in curves.items():
        vs=[r['val']['mae'] for r in h if r.get('val')]
        if vs:lines.append(f'|{method}|{fmt(min(vs))}|{fmt(vs[-1])}|')
    lines+=['','|组|协议|四轮门绝对均值|四轮输入修正幅度|M/C观测总量残差|','|---|---|---|---|---|']
    for method,e in evaluations.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];g='/'.join(fmt(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in range(1,5))
            change='/'.join(fmt(m.get(f'coe_step{i}_input_delta')) for i in range(1,5))
            res='/'.join(fmt(m.get('coe_v25_observed_sum_residual_'+s)) for s in ('M','C'))
            lines.append(f'|{method}|{label}|{g}|{change}|{res}|')
    free=evaluations.get('S01',{})
    if free.get('sets'):
        lines += ['','S01实测选择路径（最大频率路径；分布与熵见完整artifact）：','']
        for label,ev in free['sets'].items():
            paths=[(k,v) for k,v in ev['metrics'].items() if k.startswith('coe_scale_path_') and k.endswith('_fraction')]
            paths=sorted(paths,key=lambda p:-p[1])[:5]
            lines.append(f'- {label}：'+', '.join(f'{k}={v:.4f}' for k,v in paths))
    lines += ['','### 失败、缺失与审计','']
    lines += ['- '+v for v in issues] if issues else ['审计未发现配置、检查点、主指标或已完成运行的epoch序列异常。']
    for r in records:
        if not r['complete']:lines.append(f'- {r["method"]}：{r["state"]}，当前{r["epochs"]}epoch。')
        if r['failure']:lines.append(f'- {r["method"]}历史失败记录：`{r["failure"]}`；是否最终完成见上表。')
    lines += ['','## 3. 结果分析与建议','','### 预设对照的量化比较','',
        '变化率为(新/参考−1)×100%，负值表示误差下降。每种协议独立比较，不混合不同分布。','',
        '|对照|协议|MAE变化%|RMSE变化%|','|---|---|---|---|']
    pairs=[('S01','B1'),('S02','B1'),('S03','B1'),('S03','S02'),('S04','B2'),('S04','B1'),('S05','B1'),('S06','B1'),('S07','B1'),('S08','B1'),('S09','S08')]
    for a,b in pairs:
        ea=evaluations.get(a,{}).get('sets',{});eb=evaluations.get(b,{}).get('sets',{})
        for label in protocol['evaluations']:
            if label in ea and label in eb:
                ma,mb=ea[label]['metrics'],eb[label]['metrics']
                lines.append(f'|{a}/{b}|{label}|{fmt(100*(ma["mae"]/mb["mae"]-1))}|{fmt(100*(ma["rmse"]/mb["rmse"]-1))}|')
    ea=evaluations.get('S09',{}).get('sets',{});eb=evaluations.get('S08',{}).get('sets',{});ec=evaluations.get('S07',{}).get('sets',{});ed=evaluations.get('B1',{}).get('sets',{})
    for label in protocol['evaluations']:
        if all(label in e for e in (ea,eb,ec,ed)):
            same=ea[label]['metrics']['mae']-eb[label]['metrics']['mae'];cross=ec[label]['metrics']['mae']-ed[label]['metrics']['mae']
            lines.append(f'\n{label}：同尺度差分S09−S08={same:.4f}；跨尺度末轮差分S07−B1={cross:.4f}；两种配对效应之差={same-cross:.4f}。描述量，不是显著性检验。')
    lines += ['','### 证据边界与下一步','',
        '本批单训练seed7；测试mask复测不是训练种子复验。不能凭最佳值宣称稳定泛化或逐轮误差必然下降。中间轮decoder未被直接监督，逐轮预测仅作为诊断。',
        '若S04不能复现原B2负结果，先复验稳定性；若全链退步而单位置正常，优先检查累积/联合适应。若S09/S08改善而S07/B1退步，支持同尺度差分更适用的假设，但CFFF改变第二轮计算与观测重新注入，不能视为严格纯语义干预。若同尺度也退步，再检查RMS、门控可靠性或观测一致性。',
        '自由路由若主要选择单一路径，需要后续该固定路径从头训练对照。Top-k增加同时增加计算；S02/S03不等成本。所有分族负结果均应保留。成本不含所有等待及失败当轮时间；旧B2曾从GPU0迁移GPU1，新组全部GPU1，不能由设备差异推断机制因果。',
        '正式选型前对候选和对应参考补训练seed17/27及跨数据集验证；不能反复用最终OOD测试调参。MAPE遇到零值不作为主要判据。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'analysis.json',{'complete':final,'issues':issues,'records':records})
    write(suite/'report_path.json',{'path':str(path),'complete':final,'issues':issues})
    return path
