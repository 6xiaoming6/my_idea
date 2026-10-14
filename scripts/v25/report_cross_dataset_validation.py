#!/usr/bin/env python3
"""Evidence-backed staged/final report: methods, all results, then interpretation."""
import csv,json,math,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
import torch
from run_b3_c3 import load,write,digest


def fmt(x):
    return f'{x:.4f}' if isinstance(x,(int,float)) else '—'


def export_report(root,suite,js):
    evaluation_protocol = next(iter(js.values()))['config']['experiment_plan']['protocol']
    if any(j['config']['experiment_plan']['protocol'] != evaluation_protocol for j in js.values()):
        raise ValueError('Inconsistent evaluation protocols within suite')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite)
    records=[];all_results=[];families=[];issues=[];curves={};diagnostics={}
    for name,j in js.items():
        rp=suite/'results'/f'{name}.json';ep=suite/'evaluations'/f'{name}.json';pointer=suite/'runs'/f'{name}.json'
        r=load(rp) if rp.exists() else {};e=load(ep) if ep.exists() else {}
        run=Path(r['run_dir']) if r else Path(load(pointer)['run_dir']) if pointer.exists() else None
        history=[];meta={};cpinfo={}
        if run:
            hp=run/'logs/metrics.jsonl'
            if hp.exists():history=[json.loads(line) for line in hp.read_text().splitlines() if line.strip()]
            if (run/'training_metadata.json').exists():meta=load(run/'training_metadata.json')
            for cpname in ('best','last'):
                cp=run/'checkpoints'/f'{cpname}.pth'
                if cp.exists():
                    saved=torch.load(cp,map_location='cpu',weights_only=False)
                    cpinfo[cpname]={'epoch':saved['epoch'],'config_match':digest(saved['config'])==digest(j['config']),
                        'finite':all(bool(torch.isfinite(t).all()) for t in saved['model'].values() if torch.is_floating_point(t))}
            if r and (set(cpinfo)!={'best','last'} or cpinfo['last']['epoch']!=j['config']['train']['epochs'] or cpinfo['best']['epoch']!=r['best_epoch']):
                issues.append(f'{name}: checkpoint epochs/receipt mismatch')
            if r and r.get('config_sha256')!=digest(j['config']):issues.append(f'{name}: receipt config mismatch')
            if any(not x['config_match'] or not x['finite'] for x in cpinfo.values()):issues.append(f'{name}: checkpoint config/nonfinite error')
            for row in history:
                for split in ('train','val'):
                    values=row.get(split) or {}
                    for key in ('loss','mae','rmse'):
                        if key in values and (not isinstance(values[key],(int,float)) or not math.isfinite(values[key])):
                            issues.append(f'{name}: epoch {row["epoch"]} {split} {key} nonfinite')
            bad=[f'epoch {h["epoch"]} {sp}.{k}={v}' for h in history for sp in ('train','val')
                 for k,v in (h.get(sp) or {}).items() if isinstance(v,str) and v.lower() in ('nan','inf','-inf')]
            diagnostics[name]={'nonfinite_diagnostics':bad,'checkpoints':cpinfo}
        complete=(r.get('status')=='finished' and e.get('status')=='finished' and
                  set(e.get('sets',{}))==set(evaluation_protocol['evaluations']))
        if e and e.get('config_sha256')!=digest(j['config']):issues.append(f'{name}: evaluation config mismatch')
        if complete and ([h['epoch'] for h in history]!=list(range(1,j['config']['train']['epochs']+1))):
            issues.append(f'{name}: incomplete epoch history');complete=False
        if complete and (e.get('npz')!=str(Path(j['sources']['test']).resolve()) or e.get('checkpoint')!=str((run/'checkpoints/best.pth').resolve())):
            issues.append(f'{name}: test source/checkpoint mismatch');complete=False
        if complete and (e.get('protocol_sha256')!=digest(evaluation_protocol) or e.get('best_epoch')!=r['best_epoch']):
            issues.append(f'{name}: evaluation protocol/checkpoint mismatch');complete=False
        state='完成' if complete else '训练完成，评估未齐' if r else '部分训练' if history else '未开始'
        failure=suite/'failures'/f'{name}.json'
        if failure.exists():diagnostics.setdefault(name,{})['failure']=load(failure)
        row={'name':name,'method':j['dataset']+'_'+j['method'],'state':state,'run':str(run) if run else None,'epochs':history[-1]['epoch'] if history else 0,
             'best_epoch':r.get('best_epoch',cpinfo.get('best',{}).get('epoch')),'val_mae':r.get('best_val_mae'),
             'minutes':r.get('total_time_sec',sum(h.get('perf',{}).get('epoch_time_sec',0) for h in history))/60,
             'params':meta.get('total_params'),'trainable':meta.get('trainable_params'),
             'peak_gib':max((h.get('perf',{}).get('peak_memory_gb',0) for h in history),default=0),
             'amp_skips':sum(h.get('train',{}).get('amp_skipped_steps',0) for h in history),'complete':complete}
        # Keep exact scalar keys in the artifacts even if a version used different names.
        row['amp_skip_fields']={k:sum(h.get('train',{}).get(k,0) for h in history) for k in
                              {k for h in history for k in h.get('train',{}) if 'skip' in k and 'amp' in k}}
        records.append(row);curves[name]=history
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];entry={'method':j['dataset']+'_'+j['method'],'protocol':label,'mae':m['mae'],'rmse':m['rmse'],
                                  'samples':ev['samples'],'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank')}
            if not all(isinstance(entry[k],(int,float)) and math.isfinite(entry[k]) for k in ('mae','rmse')):
                issues.append(f'{name}: nonfinite test {label}')
            all_results.append(entry)
            for k,v in m.items():
                if k.startswith('coe_family_') and k.endswith('_mae'):
                    prefix=k[:-3];families.append({'method':j['dataset']+'_'+j['method'],'protocol':label,'family':k[len('coe_family_'):-4],
                                                  'mae':v,'rmse':m.get(prefix+'rmse'),'count':m.get(prefix+'count')})
            diagnostics.setdefault(name,{})[label]=m
    plan=load(suite/'plan.json') if (suite/'plan.json').exists() else {}
    planned_datasets=plan.get('datasets',['bikenyc','chap'])
    expected={f'{ds}_{m}_seed7' for ds in planned_datasets for m in ('A1','A2','B1','B2')}
    if set(js)!=expected:issues.append('Missing planned factorial cells')
    final=len(records)==len(js) and all(r['complete'] for r in records) and not issues
    phase='完整' if final else '阶段'
    dest=root/'experments_report';dest.mkdir(parents=True,exist_ok=True)
    mixed = len(evaluation_protocol['evaluations']['in_distribution']['families']) > 1
    title = 'BikeNYC_200epoch四组验证' if planned_datasets==['bikenyc'] and next(iter(js.values()))['config']['train']['epochs']==200 else 'BikeNYC_CHAP混合mask八组跨数据集验证'
    stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v25_'+title+'_'+suite.name
    assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for filename,value in [('summary.json',records),('curves.json',curves),('diagnostics.json',diagnostics),('audit.json',{'issues':issues,'complete':final})]:write(assets/filename,value)
    for filename,rows in [('protocols.csv',all_results),('families.csv',families)]:
        if rows:
            with (assets/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    datasets=list(dict.fromkeys(j['dataset'] for j in js.values()))
    fig,axes=plt.subplots(len(datasets),2,figsize=(12,4*len(datasets)),squeeze=False)
    for index,ds in enumerate(datasets):
        for n,h in curves.items():
            if not h or js[n]['dataset']!=ds:continue
            axes[index,0].plot([v['epoch'] for v in h],[v['train']['mae'] for v in h],label=js[n]['method'])
            vs=[v for v in h if v.get('val')]
            axes[index,1].plot([v['epoch'] for v in vs],[v['val']['mae'] for v in vs],label=js[n]['method'])
        for ax,title in zip(axes[index],('Train MAE (resampled masks)','Validation MAE (fixed ID masks)')):
            ax.set_title(ds+' '+title);ax.set_xlabel('Epoch');ax.set_ylabel('MAE');ax.grid(alpha=.2)
            if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=160);plt.close(fig)
    cfg=next(iter(js.values()))['config']
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}批次；仅训练 seed7。队列：`{suite}`。','',
           '## 1. 实验具体做法与复现信息','',
           '|组|输入观测|跨轮差分|对照与问题|','|---|---|---|---|',
           *[f'|{ds}_A1|仅 F，M/C 值和掩码均置零|无|单尺度输入参考|' for ds in planned_datasets],
           *[f'|{ds}_A2|仅 F，M/C 值和掩码均置零|RMS 匹配、±0.1 条件门|对同数据集 A1：差分作用|' for ds in planned_datasets],
           *[f'|{ds}_B1|F/M/C 各自稀疏观测|无|对同数据集 A1：额外尺度观测作用|' for ds in planned_datasets],
           *[f'|{ds}_B2|F/M/C 各自稀疏观测|RMS 匹配、±0.1 条件门|对同数据集 B1：差分；对 A2：额外观测|' for ds in planned_datasets],'',
           '### 数据协议与模型实现','',
           'BikeNYC：F=[B,2,12,24,12]，M=[B,2,12,12,6]，C=[B,2,12,6,3]；完整 F 按 2×2/4×4 求和，编码时 M/C 除以 4/16。CHAP 北京 PM2.5：F=[B,1,7,32,32]，M=[B,1,7,16,16]，C=[B,1,7,8,8]；区域平均构造 M/C，编码时不再除面积。都不使用验证/测试统计量归一化。',
           'BikeNYC 原始 flow_data.npy 为21×12，本地窗口为24×12，底部3行是零填充，沿用原协议参与掩码和评分，绝对误差含填充区域贡献；本批不改变此定义，四组公平比较。原切分保留，按完整帧内容哈希剔除后续集合中与前一集合重叠的窗口；不将窗口假设为连续时间轴。BikeNYC 原始 train/val 与 val/test 各重叠 6 帧；CHAP 未发现跨集重复帧。保留/剔除索引及预处理来源详见各组 data_manifest。该检查不证明原始时间戳连续性或 CHAP 既往 NaN 插值没有跨边界。',
           f'每尺度缺失率 0.4；训练族为 {cfg["data"]["train_mask_diversity"]["families"]}，均衡分配到样本并逐 epoch 重采样；验证族为 {cfg["data"]["eval_mask_diversity"]["families"]}。BikeNYC 掩码共享流入/流出通道，CHAP 为单通道，在 T/H/W 上抽样并保留四舍五入后的精确缺失数量。各尺度使用独立 RNG；四组 F 掩码完全相同。验证/测试分别在基础 mask seed 上加 20000/30000，固定掩码。多尺度组 60% 左右观测率与论文低感知率实验不同。',
           '共享观测编码器分别编码各尺度。M/C 编码在缺失位置清零后回到 F 网格，与 F 编码拼接，经 1×1 融合产生初始隐藏状态。每轮在 CMFF 对应原生分辨率上融合隐藏状态、该尺度真实观测及掩码、局部观测支持和区域中心位置；无观测处只使用推断特征作后备，不将其标记为观测。',
           '四轮共享 T/S/TD/SD/TA/ST/TL/SL 八专家，每轮仅执行原生 Top-2，每窗口共 8 次。固定 CMFF 是计算调度，动态选择的是专家。差分在共同 F 隐藏网格计算，RMS 匹配的比例 detach，门为 0.1*tanh、末层零初始化；只修正专家输入，router 读取未修正状态。',
           '每个数据集内部四组完整初始 state_dict 一致；不同通道数的数据集不强求同一 state_dict；无差分组冻结未使用门头，故总参数相同但可训练参数不同。第一批不加入历史池、乘性交互、FFN、额外视图、一致性训练、粗尺度监督或跨尺度约束损失。只以 F 隐藏位置 L1 + 0.01 candidate 均衡训练。',
           f'训练：seed7，{cfg["train"]["epochs"]} epoch，batch{cfg["data"]["batch_size"]}，每 {cfg["train"]["val_epoch"]} epoch 验证；AdamW，lr1e-3→3e-4 余弦，weight_decay1e-4，clip1，AMP，GPU1 串行。仅 ID 验证 MAE 选 best；last 每轮保存完整恢复状态。',
           f'测试 ID 族为 {evaluation_protocol["evaluations"]["in_distribution"]["families"]}；完整协议：{list(evaluation_protocol["evaluations"])}。OOD 对三个尺度施加同一类缺失形态，但独立生成各尺度掩码；不是只改变 F。已用于训练的单一结构属于 ID。mask 复测不等于训练种子复验。','',
           '### 复现与来源','',f'冻结计划：`{suite/"plan.json"}`；源码：`{suite/"source_snapshot"}`；初始化审计：`{suite/"initialization_audit.json"}`。',
           f'恢复命令：`python -u {suite/"source_snapshot/scripts/v25/run_cross_dataset_validation.py"} --suite {suite}`。完成训练跳过，缺评估补齐；失败停止。',
           '每次训练在 logs/gpu_status.jsonl 记录轻量监控。队列完成训练、评估、正式报告后退出专用 tmux。','',
           '|组|配置|运行目录|','|---|---|---|']
    if planned_datasets==['bikenyc']:
        lines=[line.replace('CHAP 北京 PM2.5：F=[B,1,7,32,32]，M=[B,1,7,16,16]，C=[B,1,7,8,8]；区域平均构造 M/C，编码时不再除面积。','').replace('，CHAP 为单通道','') for line in lines]
        lines.insert(lines.index('### 复现与来源'),'本批四组从头训练200轮，余弦周期也为200轮；相较旧100轮，训练预算与学习率轨迹同时改变，不能解释为仅追加100轮或纯训练时长因果对照。CHAP保持暂停，后续不会自动恢复。')
    lines += [f'|{r["method"]}|`{suite/"configs"/(r["name"]+".json")}`|`{r["run"]}`|' for r in records]
    lines += ['', '## 2. 实验结果','', '|组|状态|epoch|best epoch|验证 MAE|训练分钟|总参数/可训|峰值 GiB|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["state"]}|{r["epochs"]}|{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in records]
    lines += ['', '|组|测试协议|MAE|RMSE|ms/窗口|mask seed|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{r["mask_seed"]}|' for r in all_results]
    lines += ['', '### 所有缺失族结果','', '|组|协议|缺失族|MAE|RMSE|计分元素数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    lines += ['', '### 训练曲线、逐轮结果与诊断','',f'![训练与验证曲线]({assets.name}/curves.png)','',
              f'完整曲线与路由/门/观测覆盖诊断：[curves.json]({assets.name}/curves.json)、[diagnostics.json]({assets.name}/diagnostics.json)。','',
              '|组|首轮训练 MAE|末轮训练 MAE|末次验证 MAE|AMP 跳步字段|非有限诊断数量|','|---|---|---|---|---|---|']
    for r in records:
        h=curves[r['name']];vs=[v['val']['mae'] for v in h if v.get('val')]
        lines.append(f'|{r["method"]}|{fmt(h[0]["train"]["mae"] if h else None)}|{fmt(h[-1]["train"]["mae"] if h else None)}|{fmt(vs[-1] if vs else None)}|{r["amp_skip_fields"]}|{len(diagnostics.get(r["name"],{}).get("nonfinite_diagnostics",[]))}|')
    lines+=['','|组|协议|四轮 MAE|四轮门绝对值均值|M/C 物理观测残差|','|---|---|---|---|---|']
    for r in records:
        for label in evaluation_protocol['evaluations']:
            m=diagnostics.get(r['name'],{}).get(label)
            if m:
                rounds='/'.join(fmt(m.get(f'coe_x_step{i}_mae')) for i in range(1,5))
                gates='/'.join(fmt(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in range(1,5))
                residual='/'.join(fmt(m.get('coe_v25_observed_sum_residual_'+s,m.get('coe_v25_observed_mean_residual_'+s))) for s in ('M','C'))
                lines.append(f'|{r["method"]}|{label}|{rounds}|{gates}|{residual}|')
    lines+=['','M/C 残差为各 batch 观测聚合残差的均值诊断；BikeNYC 为区域总量单位，CHAP 为区域均值单位；无粗尺度观测的 A 组该值为 0，不表示满足真实粗尺度约束。','',
            '核验异常：'+('；'.join(issues) if issues else '当前已有记录未发现检查点配置、主指标非有限或完成回执冲突。缺失结果仍按阶段状态披露。')]
    for name,diag in diagnostics.items():
        if diag.get('failure'):lines+=['',f'{name} 失败记录：`{diag["failure"]}`']
        if diag.get('nonfinite_diagnostics'):lines+=['',f'{name} 非有限诊断：'+ '; '.join(diag['nonfinite_diagnostics'])]
    lines+=['','## 3. 结果分析与建议','', '### 配对结果','',
            '下表为 (新组/参考组−1)×100%，负数表示改善；不因退步省略指标。','',
            '|比较|协议|MAE 变化 %|RMSE 变化 %|','|---|---|---|---|']
    lookup={(r['method'],r['protocol']):r for r in all_results}
    for new,ref in [(ds+'_'+a,ds+'_'+b) for ds in dict.fromkeys(j['dataset'] for j in js.values()) for a,b in [('A2','A1'),('B2','B1'),('B1','A1'),('B2','A2')]]:
        for label in evaluation_protocol['evaluations']:
            a,b=lookup.get((new,label)),lookup.get((ref,label))
            if a and b:
                changes=[100*(a[k]/b[k]-1) if b[k] else float('nan') for k in ('mae','rmse')]
                lines.append(f'|{new}/{ref}|{label}|{fmt(changes[0])}|{fmt(changes[1])}|')
    for ds in dict.fromkeys(j['dataset'] for j in js.values()):
        for label in evaluation_protocol['evaluations']:
            if all((ds+'_'+m,label) in lookup for m in ('A1','A2','B1','B2')):
                a1,a2,b1,b2=(lookup[(ds+'_'+m,label)]['mae'] for m in ('A1','A2','B1','B2'))
                lines+=['',f'{ds} {label}：A2−A1={a2-a1:.4f}，B2−B1={b2-b1:.4f}；差中之差={(b2-b1)-(a2-a1):.4f}。这是单种子描述量，不是显著性检验。']
    lines+=['','### 与 TaxiBJ 的方向复验','',
            '下表报告 (B2/B1−1)×100% 的 MAE 变化，负数为改善；不跨数据集平均原始误差。仅在 B1/B2 均完成六套测试后解释方向。','',
            '|数据集|协议|B2/B1 MAE变化 %|与 TaxiBJ 同方向|','|---|---|---|---|']
    reference={}
    for method in ('A1','A2','B1','B2'):
        path=suite/'references'/method/'evaluation.json'
        if path.exists():reference[method]=load(path)
    for label in evaluation_protocol['evaluations']:
        delta=None
        if all(m in reference for m in ('B1','B2')):
            left=reference['B1']['sets'][label]['metrics']['mae'];right=reference['B2']['sets'][label]['metrics']['mae']
            delta=100*(right/left-1)
            lines.append(f'|TaxiBJ（历史冻结参考）|{label}|{fmt(delta)}|参考|')
        for ds in dict.fromkeys(j['dataset'] for j in js.values()):
            a,b=lookup.get((ds+'_B2',label)),lookup.get((ds+'_B1',label))
            if a and b:
                value=100*(a['mae']/b['mae']-1) if b['mae'] else None
                same='是' if value is not None and delta is not None and value*delta>0 else '否或零差异'
                lines.append(f'|{ds}|{label}|{fmt(value)}|{same}|')
            else:lines.append(f'|{ds}|{label}|待完成|尚不能判断|')
    lines+=['','判断重点：B1/A1 是否收益；A2/A1 与 B2/B1 的差分作用是否相反；B2 是否 ID 改善而多个 OOD 退步。差异很小只视为弱信号，不能仅凭符号认定稳定复现。原 TaxiBJ B2 曾中断并迁移 GPU，独立从头复验尚未完成，故历史结论也有运行条件限制。']
    if (suite/'references/bikenyc100').exists():
        lines+=['','### 与原100 epoch的配对比较','',
                '负数为200轮方案改善。两批数据、mask、初始化、batch和模型保持一致，200轮方案从头训练且余弦周期延长；这同时改变优化轨迹和训练预算。不是新增训练种子。','',
                '|组|协议|100轮 MAE/RMSE|200轮 MAE/RMSE|MAE变化 %|RMSE变化 %|','|---|---|---|---|---|---|']
        for method in ('A1','A2','B1','B2'):
            old=load(suite/'references/bikenyc100'/method/'evaluation.json')
            for label in evaluation_protocol['evaluations']:
                old_metrics=old['sets'][label]['metrics'];new=lookup.get(('bikenyc_'+method,label))
                old_text='/'.join(fmt(old_metrics[k]) for k in ('mae','rmse'))
                if new:
                    new_text='/'.join(fmt(new[k]) for k in ('mae','rmse'))
                    changes=[100*(new[k]/old_metrics[k]-1) for k in ('mae','rmse')]
                    lines.append(f'|{method}|{label}|{old_text}|{new_text}|{fmt(changes[0])}|{fmt(changes[1])}|')
                else:lines.append(f'|{method}|{label}|{old_text}|待完成|—|—|')
    lines+=['','### 证据边界与下一步','',
            'A1/A2 与 B1/B2 比较包含额外观测信息带来的收益，不是等信息预算的架构优越性证明。本批每尺度缺失 0.4，B 组观测标量数量约为 A 的 1+1/4+1/16=1.3125 倍；区域聚合不等于同数量细尺度观测提供的信息。',
            'B2/B1 隔离真实多尺度任务中的差分作用，A2/A1 检查该作用是否依赖多尺度观测；需逐协议看收益和负结果。单 seed7 不支持稳定泛化或统计显著性结论，测试 mask 复测也不替代多训练种子。',
            '固定 CMFF 不代表动态尺度选择；动态的是 Top-2 专家。中间轮未施加逐轮监督，不能声称误差逐轮必降。修复多尺度输入属于恢复任务正确性，不能据此宣称新贡献。',
            '下一步先检查 B2/B1 在 ID 与所有 OOD 协议的方向，再决定是否复验训练种子；后续调度对照须让 FFFF 与 CMFF 都读取同样的 F/M/C 观测。粗尺度约束损失和历史池不在本批内。',
            'A1/A2 是 v25 架构屏蔽 M/C 观测的参考，不是原封不动的 v24 K02/K04：保留新增观测融合层，C/M 轮掩码为零，后备数值与 v24 的 F 可见值覆盖聚合不同。即使混合 mask 训练与 ID/OOD 协议对齐，也不能把跨版本差异只归因于真实多尺度输入。纯随机旧批次已暂停，其检查点不作为混合 mask 批次初始化。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'summary.json',records);write(suite/'report_path.json',{'path':str(path),'complete':final,'issues':issues})
    return path
