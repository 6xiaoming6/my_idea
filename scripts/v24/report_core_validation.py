"""Formal fixed-core report: methods, all measurements, then paired conclusions."""
import json,math,os,statistics
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from run_b3_c3 import load,write,digest

PAIRS=(('K02','K01','无差分的尺度收益'),('K04','K03','有差分的尺度收益'),
       ('K03','K01','FFFF差分收益'),('K04','K02','CMFF差分收益'),
       ('K05','K02','无差分独立/共享'),('K06','K04','有差分独立/共享'),
       ('K06','K05','独立池差分'),('K07','K04','无条件/条件门'),('K08','K04','原始/RMS差分'))

def fmt(v):return f'{v:.4f}' if isinstance(v,(float,int)) and math.isfinite(v) else '—'

def export_report(root,suite,jobs):
    root,suite=Path(root),Path(suite);rows=[];curves={};diagnostics={}
    for n,j in jobs.items():
        rp=suite/'results'/f'{n}.json';ep=suite/'evaluations'/f'{n}.json';pointer=suite/'runs'/f'{n}.json'
        r=load(rp) if rp.exists() else {};e=load(ep) if ep.exists() else {}
        run=Path(r['run_dir']) if r else Path(load(pointer)['run_dir']) if pointer.exists() else None
        hp=run/'logs/metrics.jsonl' if run else None
        h=[json.loads(x) for x in hp.read_text().splitlines()] if hp and hp.exists() else [];issues=[]
        if r:
            if r.get('config_sha256')!=digest(j['config']) or r.get('completed_epochs')!=j['config']['train']['epochs']:issues.append('训练回执不匹配')
            if len(h)!=j['config']['train']['epochs']:issues.append('epoch日志不完整')
            if digest(load(run/'config.json'))!=digest(j['config']):issues.append('保存配置不一致')
            if h:
                vals=[x for x in h if x.get('val')]
                best=min(vals,key=lambda x:x['val']['mae']) if vals else None
                if not best or best['epoch']!=r['best_epoch'] or best['val']['mae']!=r['best_val_mae']:issues.append('最优验证记录不匹配')
            for k in ('best.pth','last.pth'):
                if not (run/'checkpoints'/k).exists():issues.append('缺少'+k)
        for x in h:
            for sp in ('train','val'):
                if x.get(sp) and any(not isinstance(x[sp].get(k),(int,float)) or not math.isfinite(x[sp][k]) for k in ('loss','mae','rmse')):issues.append('主指标非有限')
        sets=e.get('sets',{});protocol=j['config']['experiment_plan']['protocol']
        valid=e.get('status')=='finished' and set(sets)==set(protocol['evaluations'])
        if e and (e.get('config_sha256')!=digest(j['config']) or e.get('protocol_sha256')!=digest(protocol)):issues.append('评估回执不一致')
        for v in sets.values():
            if any(not isinstance(v['metrics'].get(k),(int,float)) or not math.isfinite(v['metrics'][k]) for k in ('mae','rmse')):issues.append('测试指标非有限')
        status='finished' if r and valid and not issues else 'trained' if r else 'partial' if h else 'pending'
        failure=suite/'failures'/f'{n}.json'
        if failure.exists() and status!='finished':status='failed';issues.append(load(failure)['error'])
        meta=load(run/'training_metadata.json') if run and (run/'training_metadata.json').exists() else {}
        rows.append(dict(variant=n,dataset=j['dataset'],method=j['method'],seed=j['config']['seed'],status=status,issues=issues,
            epoch=len(h),best_epoch=r.get('best_epoch'),val_mae=r.get('best_val_mae'),run=str(run) if run else None,
            seconds=r.get('total_time_sec'),parameters=meta.get('total_params'),trainable=meta.get('trainable_params'),
            peak_gib=max((x['perf']['peak_memory_gb'] for x in h),default=0),
            evaluations={k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in sets.items()}))
        curves[n]=h;diagnostics[n]=sets
    finished=sum(r['status']=='finished' for r in rows);final=finished==len(jobs)
    write(suite/'summary.json',rows)
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    dest=root/'experments_report'/f'{date}_v24_前两贡献主干验证_{suite.name}_{"完整分析" if final else "阶段分析"}.md'
    dest.parent.mkdir(exist_ok=True);assets=dest.with_suffix('');assets.mkdir(exist_ok=True)
    for name,value in [('metrics',rows),('curves',curves),('diagnostics',diagnostics)]:write(assets/(name+'.json'),value)
    lines=['# 固定CMFF与受控差分：前两个贡献验证','',f'完成训练及评估：{finished}/{len(jobs)}。'+('完整批次。' if final else '阶段报告，不是最终结论。'),'',
      '## 1. 具体方法与复现信息','','| 实验 | 做法 | 对照 |','| --- | --- | --- |']
    for n,j in jobs.items():lines.append(f'| {n} | {j["config"]["experiment_plan"]["method"]} | {j["reference"]} |')
    first=next(iter(jobs.values()))['config']
    lines += ['',f'共同预算：random0.4，batch{first["data"]["batch_size"]}，{first["train"]["epochs"]}epoch、每{first["train"]["val_epoch"]}epoch验证。AdamW，1e-3余弦至3e-4、wd1e-4、clip1、AMP、严格确定性、GPU0串行。best只按ID验证MAE选择，保留best/last。',
      '四轮八专家T/S/TD/SD/TA/ST/TL/SL、原生Top2和组内softmax、direct更新；固定CMFF或FFFF，无自由尺度/预热/原completion feedback。基础L1＋0.01candidate均衡，不增加视图/一致性或其他第三贡献。K05/K06独立池，其余共享。TaxiBJ F/M/C=32²/16²/8²；BikeNYC=24×12/12×6/6×3，T=12。',
      '差分取当前与上一轮输入hidden之差；按样本/通道在T/H/W上匹配当前RMS（缩放detach，分母下限1e-6），条件门为0.1tanh(MLP(两状态全局/缺失摘要))，输出层零初始化。仅改尺度投影前的专家输入，Router读取原状态；首轮无历史，不跨batch。K07门输入置零而不更换网络；K08仅取消RMS匹配。',
      '每个训练seed内公共参数逐值一致并加载冻结模板；独立池从共享池复制后独立训练，未改变公共初始化RNG。关闭TF32、benchmark，沿用W确定性池化/插值。冻结的尺度头保留，无差分组冻结门网络；K07部分参数梯度恒零，不宣称等有效容量。',
      '模型和loader使用相同训练seed，mask种子保持20260917及原split偏移。四基础族random_point/node_outage/temporal_gap/spatial_region训练逐epoch重采样，val/test固定。六套测试沿用ID、两元、形态、三元及两套mask复测，未增加训练组合。',
      f'[冻结计划]({suite/"plan.json"})；[初始化审计]({suite/"initialization_audit.json"})；[冻结完整说明]({suite/"source_snapshot/scripts/v24/README_CORE_VALIDATION.md"})。',
      f'恢复：`python -u scripts/v24/run_core_validation.py --suite {suite} --gpu 0`。','',
      '| 实验 | 配置 | 日志 |','| --- | --- | --- |']
    for r in rows:lines.append(f'| {r["variant"]} | [config]({suite/"configs"/(r["variant"]+".json")}) | '+(f'[logs]({Path(r["run"])/"logs"})' if r['run'] else '未启动')+' |')
    lines+=['','## 2. 完整结果、曲线与诊断','','| 实验 | 状态/epoch | best | val MAE | ID MAE/RMSE | 分钟 | 峰值GiB | 总/可训参数 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        m=r['evaluations'].get('in_distribution',{})
        lines.append(f'| {r["variant"]} | {r["status"]}/{r["epoch"]} | {r["best_epoch"]} | {fmt(r["val_mae"])} | {fmt(m.get("mae"))}/{fmt(m.get("rmse"))} | {fmt(r["seconds"]/60 if r["seconds"] else None)} | {fmt(r["peak_gib"])} | {r["parameters"]}/{r["trainable"]} |')
        if r['issues']:lines+=['',r['variant']+'核验问题：'+'; '.join(r['issues']),'']
    lines+=['','| 实验 | 协议 | MAE | RMSE |','| --- | --- | --- | --- |']
    for r in rows:
        for k,m in r['evaluations'].items():lines.append(f'| {r["variant"]} | {k} | {fmt(m["mae"])} | {fmt(m["rmse"])} |')
    lines+=['','| 实验 | 调用/窗口 | 面积代理 | 后三轮门幅度 |','| --- | --- | --- | --- |']
    for n,sets in diagnostics.items():
        m=sets.get('in_distribution',{}).get('metrics',{})
        gates='/'.join(fmt(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in (2,3,4))
        lines.append(f'| {n} | {fmt(m.get("coe_expert_execution_count"))} | {fmt(m.get("coe_expert_grid_equivalents"))} | {gates} |')
    lines+=['',f'[完整逐缺失族、专家路径与门诊断]({assets.name}/diagnostics.json)；[全部训练曲线数据]({assets.name}/curves.json)。']
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v24_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blocks=sorted({(j['dataset'],j['config']['seed']) for j in jobs.values()})
    for dataset,seed in blocks:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for n,j in jobs.items():
            if (j['dataset'],j['config']['seed'])!=(dataset,seed):continue
            for ax,sp in zip(axs,('train','val')):
                pts=[h for h in curves[n] if h.get(sp)]
                if pts:ax.plot([h['epoch'] for h in pts],[h[sp]['mae'] for h in pts],label=j['method'])
                ax.set_title(f'{dataset} seed{seed} {sp} MAE');ax.set_xlabel('epoch')
        for ax in axs:
            if ax.lines:ax.legend(fontsize=7)
        fig.tight_layout();name=f'{dataset}_seed{seed}.png';fig.savefig(assets/name,dpi=120);plt.close(fig)
        lines+=['',f'![{dataset} seed{seed}]({assets.name}/{name})']
    lines+=['','## 3. 配对分析、证据边界与建议','','负数为改善。多数据集分开比较，旧实验仅作背景，不混作本批对照。','','| 数据集/seed | 对照（前/后） | ID MAE变化 | ID RMSE变化 | 主要OOD MAE变化 |','| --- | --- | --- | --- | --- |']
    lookup={(r['dataset'],r['seed'],r['method']):r for r in rows};paired={}
    for dataset,seed in blocks:
        for a,b,label in PAIRS:
            ra=lookup.get((dataset,seed,a));rb=lookup.get((dataset,seed,b))
            if not ra or not rb or ra['status']!='finished' or rb['status']!='finished':continue
            changes={k:100*(m['mae']/rb['evaluations'][k]['mae']-1) for k,m in ra['evaluations'].items()}
            rmse=100*(ra['evaluations']['in_distribution']['rmse']/rb['evaluations']['in_distribution']['rmse']-1)
            ood=[k for k in ('unseen_combinations','unseen_geometry','unseen_triple') if k in changes]
            details='; '.join(f'{k}: {changes[k]:+.2f}%' for k in ood)
            lines.append(f'| {dataset}/{seed} | {a}/{b}：{label} | {changes["in_distribution"]:+.2f}% | {rmse:+.2f}% | {details} |')
            paired.setdefault((dataset,a,b),[]).append(changes['in_distribution'])
            if any(changes[k]>5 for k in ood):lines+=['',f'{dataset} seed{seed} {a}/{b} 存在主要OOD退步超过5%的泛化代价。','']
    lines+=['','| 数据集 | 对照 | 完成种子数 | 配对ID相对变化均值±样本标准差 | 改善次数 |','| --- | --- | --- | --- | --- |']
    for (dataset,a,b),values in paired.items():
        sd=fmt(statistics.stdev(values)) if len(values)>1 else '不可估'
        lines.append(f'| {dataset} | {a}/{b} | {len(values)} | {statistics.mean(values):+.4f}% ± {sd}% | {sum(v<0 for v in values)}/{len(values)} |')
    lines+=['','三个seed只是初步复验，不直接宣称统计显著；mask复测不等于训练seed复验。K05–K08仅单种子，不据此确立稳定结构优势。即使ID改善，仍须披露所有OOD代价。',
      '每窗口8次专家调用不等于同FLOPs：FFFF面积代理8，CMFF为4.625；独立池参数更多。共享若精度接近但参数/成本更低也有价值。K07与K04若相当，则条件化必要性尚未被证明；K08与K04检验RMS匹配必要性。',
      '本批未新增单层Top8、其他固定尺度路径或外部baseline，因此不能仅凭本批宣称多轮优于单层、CMFF全局最优或达到SOTA。应先确认核心四组的逐种子尺度/差分收益，再决定哪些机制能保留为贡献；结果不支持的部分如实收缩主张。第三贡献没有在本批训练或选择。']
    dest.write_text('\n'.join(lines)+'\n');(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
