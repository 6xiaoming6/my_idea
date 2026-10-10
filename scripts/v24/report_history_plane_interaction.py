"""Formal report for X01-X10 and frozen K04 reference."""
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
    dest=root/'experments_report'/f'{date}_v24_CMFF历史池三平面交互10组_{suite.name}_{"完整分析" if final else "阶段分析"}.md'
    dest.parent.mkdir(parents=True,exist_ok=True);assets=dest.with_suffix('');assets.mkdir(exist_ok=True)
    write(suite/'summary.json',records)
    for name,value in [('metrics',records),('curves',histories),('diagnostics',details)]:write(assets/(name+'.json'),value)
    lines=['# CMFF历史池、ST三平面与双专家交互','',f'批次：{suite.name}；'+('完整批次。' if final else '阶段报告，尚未全部完成。'),
      '', '## 1. 具体方法与复现信息','','| 实验 | 做法 | 对照与问题 |','|---|---|---|',
      f'| {reference} | 已完成的CMFF共享八专家、四轮原生Top-2、±0.1条件门与RMS差分；无额外FFN | 复用冻结K04 seed7结果，不新增参考训练 |']
    for n,j in payload['jobs'].items():lines.append(f'| {n} | {j["config"]["experiment_plan"]["method"]} | {j["reference"]} |')
    doc=(suite/'source_snapshot/scripts/v24/README_HISTORY_PLANE_INTERACTION.md')
    if not doc.exists():doc=Path(__file__).with_name('README_HISTORY_PLANE_INTERACTION.md')
    methods=doc.read_text().split('## 评估、报告与解释边界')[0].split('## 方法与复现设置',1)[1]
    lines += ['',methods,
      f'[冻结计划]({suite/"plan.json"})；[初始化核验]({suite/"initialization_audit.json"})；[GPU验收]({suite/"acceptance/preflight.json"})；[冻结源码]({suite/"source_snapshot"})。',
      f'恢复入口：`python -u scripts/v24/run_history_plane_interaction.py --suite {suite} --gpu 0`。完成训练跳过，从last完整恢复；本报告生成不启动训练。',
      '', '| 实验 | 配置 | 日志来源 |','|---|---|---|']
    for n,r in records.items():
        config=ref/'config.json' if n==reference else suite/'configs'/f'{n}.json'
        if not config.exists():config=suite/'plan.json'
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
    lines += ['', '| 实验 | 协议 | 轮次 | MAE | RMSE | 改善比例 | 退步比例 | 修好比例 | 破坏比例 |', '|---|---|---|---|---|---|---|---|---|']
    for n,sets in details.items():
        for protocol,v in sets.items():
            m=v['metrics']
            for step in range(1,5):
                values=[fmt(m.get(f'coe_x_step{step}_{key}')) for key in ('mae','rmse','improved_fraction','worsened_fraction','repaired_fraction','broken_fraction')]
                lines.append(f'| {n} | {protocol} | {step} | '+' | '.join(values)+' |')
    lines += ['', '逐轮比例以全部有效缺失点为分母；修好/破坏阈值为0.05×max(观测RMS,1)，仅作诊断。旧K04无新增指标显示空缺。',
      '', '| 实验 | 轮次 | 历史P1/P2/P3权重 | 历史注入幅度 | ST选择率 | TH/TW/HW权重 | 三平面幅度 | 交互幅度 |','|---|---|---|---|---|---|---|---|']
    for n,sets in details.items():
        m=sets.get('in_distribution',{}).get('metrics',{})
        for step in range(1,5):
            prefix=f'coe_x_step{step}_'
            value=lambda k:fmt(m.get(prefix+k))
            lines.append(f'| {n} | {step} | '+ '/'.join(value(f'pool_weight_P{i}') for i in (1,2,3))+' | '+value('pool_delta_abs')+' | '+value('ST_selected_fraction')+' | '+ '/'.join(value(f'plane_{i}_weight') for i in ('TH','TW','HW'))+' | '+value('plane_delta_abs')+' | '+value('interaction_abs')+' |')
    lines += ['',f'[全部曲线数据]({assets.name}/curves.json)；[历史池、三平面、交互及分族诊断]({assets.name}/diagnostics.json)。',
      '', '## 3. 配对分析、证据边界与建议',
      '', '相对变化=100×(候选/对照−1)，负数为改善；所有协议和负结果均保留。',
      '', '| 候选/对照 | 协议 | MAE变化 | RMSE变化 |','|---|---|---|---|']
    names=list(payload['jobs']); pairs=[(n,reference) for n in names]
    pairs += [(f'taxibj_{a}_seed7',f'taxibj_{b}_seed7') for a,b in [('X02','X01'),('X04','X03'),('X05','X04'),('X05','X03'),('X05','X06'),('X06','X01'),('X08','X07'),('X10','X09')]]
    for a,b in pairs:
        if records[a]['status']!='finished' or records[b]['status']!='finished':continue
        for k,v in details[a].items():
            if k not in details[b]:continue
            m=v['metrics'];refm=details[b][k]['metrics'];changes=[100*(m[z]/refm[z]-1) for z in ['mae','rmse']]
            lines.append(f'| {a}/{b} | {k} | {changes[0]:+.2f}% | {changes[1]:+.2f}% |')
            if k!='in_distribution' and any(x>5 for x in changes):lines += ['',f'**OOD退步超过5%：{a}/{b}，{k}，MAE {changes[0]:+.2f}%、RMSE {changes[1]:+.2f}%。**','']
        id_change=100*(details[a]['in_distribution']['metrics']['mae']/details[b]['in_distribution']['metrics']['mae']-1)
        lines += ['',f'{a}/{b} 的ID MAE变化为{id_change:+.2f}%；'+'该差值需结合三类OOD与成本判断。'+'不能据单seed宣称稳定收益。']
    lines += ['', '| 实验 | 相对X01 ID MAE变化 | 三类OOD平均MAE变化 | 最差主要OOD变化 |', '|---|---|---|---|']
    base=details.get('taxibj_X01_seed7',{});primary=('unseen_combinations','unseen_geometry','unseen_triple')
    for n,sets in details.items():
        if all(k in sets and k in base for k in ('in_distribution',*primary)):
            d=lambda k:100*(sets[k]['metrics']['mae']/base[k]['metrics']['mae']-1)
            changes=[d(k) for k in primary]
            lines.append(f'| {n} | {d("in_distribution"):+.2f}% | {sum(changes)/3:+.2f}% | {max(changes):+.2f}% |')
    oldpath=ref/'K02_evaluations.json'
    if oldpath.exists() and records.get('taxibj_X01_seed7',{}).get('status')=='finished':
        old=load(oldpath)['sets'];lines += ['', 'X01与旧K02复现核对（所有测试，原始MAE/RMSE差值）：']
        for k,v in base.items():
            if k in old:lines.append(f'- {k}: MAE {v["metrics"]["mae"]-old[k]["metrics"]["mae"]:+.10f}, RMSE {v["metrics"]["rmse"]-old[k]["metrics"]["rmse"]:+.10f}')
    lines += ['', '分析口径：先分别检查各候选对直接控制的差值，再以X01/K04作为统一参考；三类OOD平均是相对变化的等权平均，两套mask复测单列，不重复加权。不设事后ID容忍阈值，不用测试重新选择checkpoint。',
      'X04/X03检验完整历史，X05/X04检验自适应读取（评分器额外161参数），X05/X06隔离差分。X08/X07严格匹配三平面模块参数，仅改覆盖融合；X07/K04仍混有容量差异。X10/X09各6240新增参数，但乘积算子额外成本不能省略。只看相对K04改善不足以宣称超过无差分X01。',
      '本批只有TaxiBJ seed7，mask复测不等于训练种子复验；不宣称稳定泛化或第三贡献已证实。固定CMFF不等于动态尺度；逐轮状态变化、注意权重或改善点比例不保证每轮误差下降。三平面覆盖是观测支持，不能直接解释为校准置信度。',
      '下一步根据完整ID/OOD/成本取舍选择值得独立确认的方向；补训练seed与跨数据集、必要的等容量3D对照及新确认协议。未完成时继续冻结十组，不按阶段测试修改训练方案。']
    dest.write_text('\n'.join(lines)+'\n');(suite/'report_path.txt').write_text(str(dest)+'\n')
    return dest
