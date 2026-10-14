#!/usr/bin/env python3
"""Rate-separated, complete-evidence report for BikeNYC S01-S09."""
import csv,json,math,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from report_scale_delta_exploration import collect,fmt

def key(rate,method):return f'r{int(rate*10):02d}_{method}'

def export_report(root,suite,js):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite);records=[];curves={};evals={};issues=[];results=[];families=[]
    protocols={}
    for n,j in js.items():
        rate=j['rate'];p=j['config']['experiment_plan']['protocol'];k=key(rate,j['method'])
        if p['rate']!=rate or j['config']['data']['mask']['missing_rate']!=rate:issues.append(n+': rate mismatch')
        if rate in protocols and protocols[rate]!=p:issues.append(n+': inconsistent within-rate protocol')
        protocols[rate]=p
        row,h,e,bad=collect(suite,n,j);row.update(rate=rate,label=k);records.append(row);curves[k]=h;evals[k]=e;issues+=bad
        for label,ev in e.get('sets',{}).items():
            if ev.get('mask_spec',{}).get('rates')!=[rate]:issues.append(n+': wrong evaluation rate '+label)
    expected={f'bikenyc_r{r:02d}_S{i:02d}_seed7' for r in (4,6) for i in range(1,10)}|{'bikenyc_r06_B1_seed7'}
    if set(js)!=expected:issues.append('Missing expected 19-job set')
    references={}
    for method in ('B1','B2'):
        d=suite/'references'/method;k=key(.4,method)
        if not (d/'evaluation.json').exists():issues.append('Missing .4 reference '+method);continue
        e,r,c=load(d/'evaluation.json'),load(d/'receipt.json'),load(d/'config.json')
        p=protocols.get(.4)
        if (e.get('status')!='finished' or e.get('protocol_sha256')!=digest(p) or
            set(e.get('sets',{}))!=set(p['evaluations']) or e.get('config_sha256')!=digest(c) or
            r.get('config_sha256')!=digest(c) or r.get('completed_epochs')!=c['train']['epochs']):issues.append('Invalid .4 reference '+method)
        references[k]=r;evals[k]=e;curves[k]=[json.loads(x) for x in (d/'history.json').read_text().splitlines() if x.strip()]
    for k,e in evals.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics']
            if not all(isinstance(m.get(x),(int,float)) and math.isfinite(m[x]) for x in ('mae','rmse')):issues.append(k+': nonfinite test')
            results.append({'method':k,'protocol':label,'mae':m['mae'],'rmse':m['rmse'],'samples':ev['samples'],
                'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank'),
                'expert_calls':m.get('coe_expert_execution_count'),'area':m.get('coe_expert_grid_equivalents')})
            for name,value in m.items():
                if name.startswith('coe_family_') and name.endswith('_mae'):
                    families.append({'method':k,'protocol':label,'family':name[len('coe_family_'):-4],'mae':value,
                        'rmse':m.get(name[:-3]+'rmse'),'count':m.get(name[:-3]+'count')})
    final=not issues and len(records)==19 and all(r['complete'] for r in records)
    phase='完整' if final else '阶段';stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v25_BikeNYC双缺失率S01-S09探索_'+suite.name
    dest=root/'experments_report';dest.mkdir(parents=True,exist_ok=True);assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for name,value in [('summary.json',records),('curves.json',curves),('evaluations.json',evals),('audit.json',{'complete':final,'issues':issues})]:write(assets/name,value)
    for name,rows in [('protocols.csv',results),('families.csv',families)]:
        if rows:
            with (assets/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,rate in enumerate((.4,.6)):
        for k,h in curves.items():
            if not h or not k.startswith(f'r{int(rate*10):02d}_'):continue
            axes[i,0].plot([r['epoch'] for r in h],[r['train']['mae'] for r in h],label=k)
            vals=[r for r in h if r.get('val')];axes[i,1].plot([r['epoch'] for r in vals],[r['val']['mae'] for r in vals],label=k)
        for ax,name in zip(axes[i],('Train MAE','ID val MAE')):
            ax.set(title=f'rate {rate}: {name}',xlabel='Epoch',ylabel='MAE');ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=160);plt.close(fig)
    cfg=next(iter(js.values()))['config']
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}，seed7；18组S实验＋必要B1@0.6，共19次。队列 `{suite}`。','',
        '## 1. 实验具体做法与复现信息','',
        '|缺失率|组|尺度路径|逐轮Top-k|差分输入轮次|具体做法与对照|','|---|---|---|---|---|---|']
    for j in js.values():
        s=j['config']['model']['coe']['scale_delta_exploration']
        lines.append(f'|{j["rate"]}|{j["method"]}|{s["path"]}|{s["top_k_per_step"]}|{s["delta_steps"]}|{j["config"]["experiment_plan"]["method"]}；对照 {j["reference"]}|')
    lines+=['','B1/B2@0.4来自旧200轮套件的冻结参考，S04@0.4按请求从头复验；相同seed重复不是多种子。0.6新增B1作为公平参考，S04作为完整差分组，未借用0.4得分作为0.6基准。','',
        'BikeNYC窗口F=[B,2,12,24,12]、M=[B,2,12,12,6]、C=[B,2,12,6,3]。区域求和构造完整M/C后独立抽观测mask，编码时除已知面积4/16。H0融合三尺度编码，每轮再注入对应尺度可见数据；没有粗尺度或中间轮监督。模型输出为F形状补全预测，损失仅F隐藏位置L1＋0.01 candidate均衡。',
        '沿用清理跨split重复帧后的511/72/146窗口。BikeNYC原始21×12网格补成24×12，底部3行仍参与mask与评分。跨率/组均沿用该定义；不能用低绝对误差声称跨数据集更难任务被解决。',
        f'训练{cfg["train"]["epochs"]}epoch、batch{cfg["data"]["batch_size"]}、val每{cfg["train"]["val_epoch"]}轮、seed7；AdamW 1e-3余弦至3e-4、weight_decay1e-4、clip1、AMP、严格确定性。0.4/0.6各自训练/验证/六套测试均为本率。四基础族混合、逐epoch重采样，ID val MAE选best，best/last保存。所有组初始化核验为旧B1同一完整state_dict。',
        '六套测试为ID基础族、未见两元组合、移动/多块几何、三元组合、两元mask复测、几何mask复测；种子沿用冻结协议，val/test分别偏移20000/30000。测试mask复测不是训练seed复验。完整配置见下表。',
        '四轮共享T/S/TD/SD/TA/ST/TL/SL八专家。只执行选中专家与单一选中尺度；Top-k集合内softmax。S01使用硬选择尺度和选中分支ST近似梯度，S03 Top1训练系数也使用ST、前向为1。无软尺度预热、FFN、历史池、乘积交互、第二视图、一致性。K0数值后备固定，各轮状态直接传递。',
        '受控差分H当前−H上一轮，按样本通道RMS匹配（比例detach）并乘±0.1条件门，仅修正专家输入，router读取原状态。首轮无差分。CMFF的末轮差分H3−H2来自F/M输出；CFFF末轮来自两个F输出。单位置组保留历史更新，只限制差分启用轮次。',
        '每窗口调用次数：S02=16，S03=15，其余8。Top-k扩展改变计算预算，可训参数也随门/自由尺度头启用变化，不宣称等成本。','',
        f'冻结源码 `{suite/"source_snapshot"}`；计划 `{suite/"plan.json"}`；初始化 `{suite/"initialization_audit.json"}`。',
        f'恢复命令：`python -u {suite/"source_snapshot/scripts/v25/run_bikenyc_scale_delta_rates.py"} --suite {suite} --gpu 1`。GPU1串行，logs/gpu_status.jsonl每15秒采样，专用tmux在训练/评估/报告全部完成后退出。CHAP与旧TaxiBJ保持暂停。','',
        '|组|配置|运行目录|','|---|---|---|']
    lines += [f'|{r["label"]}|`{suite/"configs"/(r["name"]+".json")}`|`{r["run"]}`|' for r in records]
    lines += [f'|旧{k}|`{suite/"references"/k.split("_")[-1]/"config.json"}`|`{r["run_dir"]}`|' for k,r in references.items()]
    lines += ['','## 2. 实验结果','','|组|状态|已完成epoch|best epoch|val MAE|分钟|总参数/可训|峰值GiB|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["label"]}|{r["state"]}|{r["epochs"]}|{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in records]
    lines += ['','|组|测试协议|MAE|RMSE|ms/窗口|专家次数|面积代理|','|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{fmt(r["expert_calls"])}|{fmt(r["area"])}|' for r in results]
    lines += ['','### 分缺失族结果','','|组|协议|缺失族|MAE|RMSE|计分元素数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    lines += ['','### 曲线与诊断','',f'![训练与验证曲线]({assets.name}/curves.png)','',
        f'[完整曲线]({assets.name}/curves.json)；[完整评估与路由、逐轮、RMS、观测残差诊断]({assets.name}/evaluations.json)。','',
        '|组|首轮/末轮train MAE|best/末次val MAE|','|---|---|---|']
    for k,h in curves.items():
        if h:
            vs=[r['val']['mae'] for r in h if r.get('val')]
            lines.append(f'|{k}|{fmt(h[0]["train"]["mae"])}/{fmt(h[-1]["train"]["mae"])}|{fmt(min(vs) if vs else None)}/{fmt(vs[-1] if vs else None)}|')
    lines += ['','|组|协议|四轮门绝对均值|四轮输入修正幅度|M/C观测总量残差|','|---|---|---|---|---|']
    for k,e in evals.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];g='/'.join(fmt(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in range(1,5));d='/'.join(fmt(m.get(f'coe_step{i}_input_delta')) for i in range(1,5));res='/'.join(fmt(m.get('coe_v25_observed_sum_residual_'+n)) for n in ('M','C'))
            lines.append(f'|{k}|{label}|{g}|{d}|{res}|')
    lines += ['','### 审计、失败与缺失','']
    lines += ['- '+x for x in issues] if issues else ['当前已存在证据未发现配置/主指标/检查点异常；完成状态以上表为准。']
    for r in records:
        if not r['complete']:lines.append(f'- {r["label"]}：{r["state"]}。')
        if r['failure']:lines.append(f'- {r["label"]}历史失败：`{r["failure"]}`。')
    lines += ['','## 3. 结果分析与建议','','### 同缺失率预设对照','','负值为误差改善；不同缺失率独立比较。','','|缺失率|对照|协议|MAE变化%|RMSE变化%|','|---|---|---|---|---|']
    pairs=[('S01','B1'),('S02','B1'),('S03','B1'),('S03','S02'),('S04','B2'),('S04','B1'),('S05','B1'),('S06','B1'),('S07','B1'),('S05','S04'),('S06','S04'),('S07','S04'),('S08','B1'),('S09','S08')]
    for rate,p in protocols.items():
        for a,b in pairs:
            ea=evals.get(key(rate,a),{}).get('sets',{});eb=evals.get(key(rate,b),{}).get('sets',{})
            for label in p['evaluations']:
                if label in ea and label in eb:
                    ma,mb=ea[label]['metrics'],eb[label]['metrics'];change=[100*(ma[x]/mb[x]-1) if mb[x] else None for x in ('mae','rmse')]
                    lines.append(f'|{rate}|{a}/{b}|{label}|{fmt(change[0])}|{fmt(change[1])}|')
        for label in p['evaluations']:
            sets=[evals.get(key(rate,m),{}).get('sets',{}) for m in ('S09','S08','S07','B1')]
            if all(label in e for e in sets):
                a,b,c,d=[e[label]['metrics']['mae'] for e in sets]
                lines.append(f'\nrate{rate} {label}：同尺度配对S09−S08={a-b:.4f}；跨尺度配对S07−B1={c-d:.4f}；效应差={(a-b)-(c-d):.4f}。单种子描述量，不是显著性检验。')
    lines += ['','### 证据边界及下一步','',
        '优先看同率S04/B1，以及单位置组/B1；S04@0.4是相同seed的实现/运行复验，不能增加统计样本量。单seed且大量候选，最佳结果需要独立seed17/27确认。0.4/0.6并非训练种子复验，测试mask复测也不是。',
        'S09/S08与S07/B1可筛选同尺度差分假设，但CFFF第二轮计算与观测重新注入一起改变，不能归因为纯差分语义。若单位置好于全链，仍需独立验证累积或联合适应假设。中间轮未监督，不声称逐轮误差必降。',
        'S01若集中某一路径，需要固定该路径从头训练对照；增加Top-k的收益需同时披露成本。0.6可能有更大差异，也可能更不稳定，不能只选择有利缺失率报告。不要反复依赖最终OOD测试调参；正式方案需跨训练种子、跨数据集确认。',
        'GPU未要求独占，计时不等于严格性能基准；所有方法使用同一已清理数据与零填充评分范围。MAPE受大量零值影响，不作为主要判据。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'analysis.json',{'complete':final,'issues':issues,'records':records});write(suite/'report_path.json',{'path':str(path),'complete':final,'issues':issues})
    return path
