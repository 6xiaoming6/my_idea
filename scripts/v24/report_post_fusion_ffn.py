"""Automatic formal report for two post-fusion FFNs and frozen K04 reference."""
import math
import os
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from run_b3_c3 import load, write, digest


def fmt(x):
    return f'{x:.4f}' if isinstance(x,(float,int)) and math.isfinite(x) else '—'


def export_report(root,suite,payload):
    root,suite=Path(root),Path(suite);records={};histories={};details={}
    ref=suite/'reference';reference=payload['reference']['name']
    definitions={reference:{'config':load(ref/'config.json'),'method':'K04'}}
    definitions.update(payload['jobs'])
    for name,j in definitions.items():
        baseline=name==reference
        rp=ref/'result.json' if baseline else suite/'results'/f'{name}.json'
        ep=ref/'evaluations.json' if baseline else suite/'evaluations'/f'{name}.json'
        r=load(rp) if rp.exists() else {};e=load(ep) if ep.exists() else {}
        pointer=suite/'runs'/f'{name}.json'
        run=Path(r['run_dir']) if r else Path(load(pointer)['run_dir']) if pointer.exists() else None
        hp=ref/'metrics.jsonl' if baseline else run/'logs/metrics.jsonl' if run else None
        h=[json.loads(x) for x in hp.read_text().splitlines()] if hp and hp.exists() else []
        mp=ref/'training_metadata.json' if baseline else run/'training_metadata.json' if run else None
        meta=load(mp) if mp and mp.exists() else {};issues=[];protocol=j['config']['experiment_plan']['protocol']
        if r:
            if r.get('status')!='finished' or r.get('completed_epochs')!=j['config']['train']['epochs'] or r.get('config_sha256')!=digest(j['config']):issues.append('训练回执不匹配')
            if len(h)!=j['config']['train']['epochs']:issues.append('epoch日志不完整')
            vals=[x for x in h if x.get('val')]
            best=min(vals,key=lambda x:x['val']['mae']) if vals else None
            if not best or (best['epoch'],best['val']['mae'])!=(r['best_epoch'],r['best_val_mae']):issues.append('best选择不匹配')
            if run:
                if digest(load(run/'config.json'))!=digest(j['config']):issues.append('保存配置不匹配')
                for cp in ['best.pth','last.pth']:
                    if not (run/'checkpoints'/cp).exists():issues.append('缺少'+cp)
        for x in h:
            for sp in ['train','val']:
                if x.get(sp) and any(not isinstance(x[sp].get(k),(int,float)) or not math.isfinite(x[sp][k]) for k in ['loss','mae','rmse']):issues.append('训练主指标非有限')
        if e and (e.get('config_sha256')!=digest(j['config']) or e.get('protocol_sha256')!=digest(protocol)):issues.append('评估指纹不匹配')
        if e and r and (e.get('best_epoch')!=r['best_epoch'] or e.get('checkpoint')!=str(run/'checkpoints/best.pth')):issues.append('评估checkpoint不匹配')
        for a in e.get('sets',{}).values():
            if any(not isinstance(a['metrics'].get(k),(int,float)) or not math.isfinite(a['metrics'][k]) for k in ['mae','rmse']):issues.append('评估指标非有限')
        finished=bool(r) and e.get('status')=='finished' and set(e.get('sets',{}))==set(protocol['evaluations']) and not issues
        failure=suite/'failures'/f'{name}.json'
        if failure.exists() and not finished:issues.append(load(failure)['error'])
        records[name]={'status':'finished' if finished else 'failed' if issues else 'partial' if h else 'pending',
            'epoch':len(h),'best_epoch':r.get('best_epoch'),'val_mae':r.get('best_val_mae'),
            'minutes':r.get('total_time_sec',0)/60 if r else None,'parameters':meta.get('total_params'),
            'trainable':meta.get('trainable_params'),'peak_gib':max((x['perf']['peak_memory_gb'] for x in h),default=0),
            'run':str(run) if run else None,'issues':issues,'evaluations':e.get('sets',{})}
        histories[name]=h;details[name]=e.get('sets',{})
    final=all(records[n]['status']=='finished' for n in definitions)
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    dest=root/'experments_report'/f'{date}_v24_CMFF受控差分_FFN共享对照_{suite.name}_{"完整分析" if final else "阶段分析"}.md'
    dest.parent.mkdir(parents=True,exist_ok=True);assets=dest.with_suffix('');assets.mkdir(exist_ok=True)
    write(suite/'summary.json',records)
    for name,value in [('metrics',records),('curves',histories),('diagnostics',details)]:write(assets/(name+'.json'),value)
    lines=['# CMFF受控差分：每轮融合后4倍FFN共享对照','',f'批次：{suite.name}；'+('完整批次。' if final else '阶段报告，尚未全部完成。'),
      '', '## 1. 具体方法与复现信息','','| 实验 | 做法 | 对照与问题 |','|---|---|---|',
      f'| {reference} | 已完成的CMFF共享八专家、四轮原生Top-2、±0.1条件门与RMS差分；无额外FFN | 复用冻结K04 seed7结果，不新增第三次训练 |']
    for n,j in payload['jobs'].items():lines.append(f'| {n} | {j["config"]["experiment_plan"]["method"]} | K04：加入FFN；F02/F01：仅改变FFN是否跨轮共享 |')
    lines += ['', '每轮：原CMFF专家执行→两专家加权融合→恢复细网格U→Hnext=U+W2(GELU(W1(LN(U))))→解码/传下一轮。维度64→256→64，逐时空位置、Pre-LN、残差相加、无dropout；LN按通道，线性层用1×1×1 Conv3d实现。四轮均执行全分辨率FFN（B×64×12×32×32），不是每个专家各自加FFN。历史仍保存上一轮输入；下一轮受控差分基于含FFN的新隐藏状态。',
      'F01只有一个FFN模块复用四次；F02有四个参数独立模块，每个从F01同一初始FFN逐值复制。新增模块使用隔离seed+94001随机流、PyTorch默认非零线性初始化。公共骨干逐值校验并加载旧K04 seed7初始化，模型/loader随机流不因FFN新增而偏移。F01/F02初始前向相同；相同结构预算但参数量不同。',
      'TaxiBJ random0.4、seed7、100epoch、batch32、GPU0串行、每5epoch验证。AdamW，lr1e-3余弦到3e-4、wd1e-4、clip1、AMP、严格确定性、无早停。L1＋0.01candidate均衡，最佳checkpoint只按ID验证MAE，保存best/last；没有额外视图、一致性或自由尺度。基础四族random_point/node_outage/temporal_gap/spatial_region近似均衡、训练逐epoch重采样，验证固定。输入[B,2,12,32,32]，隐藏64通道，F/M/C=32²/16²/8²；T/S/TD/SD/TA/ST/TL/SL共享池和组内softmax保持K04实现。',
      '训练mask基础seed20260917，验证+20000；六套测试ID/两元/形态/三元/两元mask复测/形态mask复测，基础seed依次20260917/18/19/30、20261001/02，测试+30000。相同测试窗口和mask；两套复测不是新增训练种子。有效缺失点汇总MAE/RMSE；MAPE受零或近零流量影响，不作为主要结论。',
      '每窗口仍8次专家调用、专家面积代理4.625；另有4次全分辨率FFN。额外FFN主线性层约1.611G MAC/窗口（不含LN/GELU/bias），不能把原面积代理当成总模型FLOPs。一个FFN新增33216参数；四个新增132864，共享不减少FFN执行次数。参数和实测时间/显存分开报告。',
      f'[冻结计划]({suite/"plan.json"})；[初始化核验]({suite/"initialization_audit.json"})；[GPU验收]({suite/"acceptance/preflight.json"})；[冻结源码]({suite/"source_snapshot"})；[K04参考快照]({ref})。原主线继续CMFF受控差分；其跨seed与OOD局限沿用核心28组报告，不因保留主线而改变证据判断。',
      f'复现入口：`python -u scripts/v24/run_post_fusion_ffn.py --suite {suite} --gpu 0`。完成训练跳过、缺评估单独补齐、中断从last恢复；本报告生成不启动训练。',
      '', '| 实验 | 配置 | 日志来源 |','|---|---|---|']
    for n,r in records.items():
        config=ref/'config.json' if n==reference else suite/'configs'/f'{n}.json'
        lines.append(f'| {n} | [config]({config}) | '+(f'[run]({r["run"]})' if r['run'] else '尚未启动；配置定义见冻结plan')+' |')
    lines += ['', '## 2. 完整结果、曲线与诊断','','| 实验 | 状态/epoch | best | val MAE | ID MAE/RMSE | 分钟 | 峰值GiB | 总/可训参数 | ID ms/窗口 |','|---|---|---|---|---|---|---|---|---|']
    for n,r in records.items():
        m=details[n].get('in_distribution',{}).get('metrics',{})
        lines.append(f'| {n} | {r["status"]}/{r["epoch"]} | {r["best_epoch"]} | {fmt(r["val_mae"])} | {fmt(m.get("mae"))}/{fmt(m.get("rmse"))} | {fmt(r["minutes"])} | {fmt(r["peak_gib"])} | {r["parameters"]}/{r["trainable"]} | {fmt(m.get("forward_ms_per_sample_per_rank"))} |')
        if r['issues']:lines += ['',n+'问题：'+'；'.join(r['issues']),'']
    lines += ['', '分钟为epoch内训练＋验证时间，不含checkpoint落盘、启动及六套评估；GiB为最大allocated显存；前向耗时为本次ID测试测量，非独立多次性能基准。',
      '', '| 实验 | 协议 | MAE | RMSE |','|---|---|---|---|']
    for n,sets in details.items():
        for k,v in sets.items():lines.append(f'| {n} | {k} | {fmt(v["metrics"]["mae"])} | {fmt(v["metrics"]["rmse"])} |')
    lines += ['', '| 实验 | 协议 | 缺失族 | MAE | RMSE |','|---|---|---|---|---|']
    for n,sets in details.items():
        for k,v in sets.items():
            m=v['metrics']
            for key in sorted(m):
                if key.startswith('coe_family_') and key.endswith('_mae'):
                    fam=key[len('coe_family_'):-4];lines.append(f'| {n} | {k} | {fam} | {fmt(m[key])} | {fmt(m.get(key[:-3]+"rmse"))} |')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v24_ffn_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for n,h in histories.items():
        for ax,(sp,metric) in zip(axs.flat,[('train','loss'),('train','mae'),('val','mae'),('val','rmse')]):
            points=[x for x in h if x.get(sp)]
            if points:ax.plot([x['epoch'] for x in points],[x[sp][metric] for x in points],label=n)
            ax.set_title(f'{sp} {metric}');ax.set_xlabel('epoch');ax.grid(alpha=.2)
    for ax in axs.flat:
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=150);plt.close(fig)
    lines += ['',f'![训练验证曲线]({assets.name}/curves.png)',
      '', '| 实验 | train loss 首/末 | train MAE 首/末 | val MAE 首/末 | AMP跳步 |','|---|---|---|---|---|']
    for n,h in histories.items():
        if not h:continue
        v=[x for x in h if x.get('val')]
        vm=f'{fmt(v[0]["val"]["mae"])}/{fmt(v[-1]["val"]["mae"])}' if v else '尚无验证'
        lines.append(f'| {n} | {fmt(h[0]["train"]["loss"])}/{fmt(h[-1]["train"]["loss"])} | {fmt(h[0]["train"]["mae"])}/{fmt(h[-1]["train"]["mae"])} | {vm} | {sum(x["train"].get("train_skipped_amp_steps",0) for x in h):.0f} |')
    lines += ['', '| 实验 | 专家对路径数/最大占比 | 第2–4轮门绝对均值 | 第1–4轮FFN平均绝对改变量 |','|---|---|---|---|']
    for n,sets in details.items():
        m=sets.get('in_distribution',{}).get('metrics',{})
        lines.append(f'| {n} | {fmt(m.get("coe_pair_path_unique_count"))}/{fmt(m.get("coe_pair_path_max_fraction"))} | '+ '/'.join(fmt(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in [2,3,4])+' | '+ '/'.join(fmt(m.get(f'coe_post_ffn_step{i}_change_abs')) for i in [1,2,3,4])+' |')
    lines += ['',f'[全部曲线数据]({assets.name}/curves.json)；[全部路径、门、FFN及分族诊断]({assets.name}/diagnostics.json)。',
      '', '## 3. 配对分析、证据边界与建议',
      '', '相对变化=100×(候选/对照−1)，负数为改善；所有协议和负结果均保留。',
      '', '| 候选/对照 | 协议 | MAE变化 | RMSE变化 |','|---|---|---|---|']
    names=list(payload['jobs']);pairs=[(n,reference) for n in names]+[(names[1],names[0])]
    for a,b in pairs:
        if records[a]['status']!='finished' or records[b]['status']!='finished':continue
        for k,v in details[a].items():
            if k not in details[b]:continue
            m=v['metrics'];refm=details[b][k]['metrics'];changes=[100*(m[z]/refm[z]-1) for z in ['mae','rmse']]
            lines.append(f'| {a}/{b} | {k} | {changes[0]:+.2f}% | {changes[1]:+.2f}% |')
            if k!='in_distribution' and any(x>5 for x in changes):lines += ['',f'**OOD退步超过5%：{a}/{b}，{k}，MAE {changes[0]:+.2f}%、RMSE {changes[1]:+.2f}%。**','']
        id_change=100*(details[a]['in_distribution']['metrics']['mae']/details[b]['in_distribution']['metrics']['mae']-1)
        lines += ['',f'{a}/{b} 的ID MAE变化为{id_change:+.2f}%；'+('达到单seed ≥1%改善信号。' if id_change<=-1 else '未达到单seed ≥1%改善信号。')+'不能据单seed宣称稳定收益。']
    lines += ['', 'F01/F02使用同一FFN初值、公共骨干及mask，但独立组参数更多；相同FFN调用数不等于相同参数容量。相对K04的提升同时包含新增归一化、非线性、残差分支和容量，不能只归因于4倍升维。F02/F01检验整组共享取舍，不保证解耦所有优化因素。',
      '本批只有seed7；mask复测不等于训练种子复验。固定CMFF与动态专家选择严格区分；逐轮状态变化/门幅度不证明误差必降。新增FFN不自动成为第三项核心贡献，也不能用FFN后的整体精度掩盖原差分的独立消融结论。',
      '下一步先检查ID验证/测试方向、逐协议OOD以及成本；有收益再补独立训练seed，必要时与同容量替代或CMFF无差分＋同FFN进行独立对照。未完成时继续既定两组，不依据阶段测试改变候选或超参数。']
    dest.write_text('\n'.join(lines)+'\n');(suite/'report_path.txt').write_text(str(dest)+'\n')
    return dest
