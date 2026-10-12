"""Evidence-audited Y01-Y12 report, with all original X controls disclosed."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from run_b3_c3 import load, write, digest
from run_scale_rate_compare import filehash

PRIMARY = ('unseen_combinations', 'unseen_geometry', 'unseen_triple')
PAIRS = [('Y01','X01'),('Y02','K04'),('Y03','X06'),('Y04','X05'),
         ('Y02','Y01'),('Y03','Y01'),('Y04','Y02'),('Y04','Y03'),
         ('Y06','Y05'),('Y07','Y05'),('Y08','Y06'),('Y08','Y07'),
         ('Y09','K04'),('Y10','Y09'),('Y09','X07'),('Y10','X08'),
         ('Y11','X05'),('Y11','Y10'),('Y12','X05'),('Y12','X10')]


def fmt(value):
    return f'{value:.4f}' if isinstance(value, (int,float)) and math.isfinite(value) else '—'


def collect(suite, label, job, reference=False):
    name = job['variant']; config = job['config']; protocol = config['experiment_plan']['protocol']
    folder = suite/'references'/label
    rp = folder/'result.json' if reference else suite/'results'/f'{name}.json'
    ep = folder/'evaluations.json' if reference else suite/'evaluations'/f'{name}.json'
    receipt = load(rp) if rp.exists() else {}; evaluation = load(ep) if ep.exists() else {}
    pointer = suite/'runs'/f'{name}.json'
    run = Path(receipt['run_dir']) if receipt else Path(load(pointer)['run_dir']) if pointer.exists() and not reference else None
    hp = folder/'metrics.jsonl' if reference else run/'logs/metrics.jsonl' if run else None
    issues = []; history = []
    if hp and hp.exists():
        for line in hp.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                issues.append('损坏或未写完的epoch日志')
    metadata_path = folder/'training_metadata.json' if reference else run/'training_metadata.json' if run else None
    meta = load(metadata_path) if metadata_path and metadata_path.exists() else {}
    if receipt:
        if receipt.get('status') != 'finished' or receipt.get('completed_epochs') != config['train']['epochs'] or receipt.get('config_sha256') != digest(config):
            issues.append('训练回执与冻结配置不匹配')
        if [r['epoch'] for r in history] != list(range(1, config['train']['epochs']+1)):
            issues.append('epoch序列不完整')
        vals = [r for r in history if r.get('val')]
        best = min(vals, key=lambda r:r['val']['mae']) if vals else None
        if not best or (best['epoch'],best['val']['mae']) != (receipt.get('best_epoch'),receipt.get('best_val_mae')):
            issues.append('best验证选择不匹配')
        cp_config = folder/'config.json' if reference else run/'config.json'
        if not cp_config.exists() or digest(load(cp_config)) != digest(config):
            issues.append('保存配置不匹配')
        for filename in ('best.pth','last.pth'):
            if not (run/'checkpoints'/filename).exists():
                issues.append('缺少'+filename)
    for row in history:
        for split in ('train','val'):
            if row.get(split) and any(not isinstance(row[split].get(k),(int,float)) or not math.isfinite(row[split][k]) for k in ('loss','mae','rmse')):
                issues.append('训练主指标非有限')
    if evaluation:
        if evaluation.get('config_sha256') != digest(config) or evaluation.get('protocol_sha256') != digest(protocol):
            issues.append('评估配置或协议指纹不匹配')
        if receipt and (evaluation.get('best_epoch') != receipt.get('best_epoch') or evaluation.get('checkpoint') != str(run/'checkpoints/best.pth')):
            issues.append('评估checkpoint不匹配')
        if evaluation.get('split', 'test') != 'test' or evaluation.get('npz') != str(Path(job['sources']['test']).resolve()):
            issues.append('评估数据来源不匹配')
        for key, value in evaluation.get('sets', {}).items():
            m = value['metrics']
            if any(not isinstance(m.get(k),(int,float)) or not math.isfinite(m[k]) for k in ('mae','rmse')):
                issues.append('测试主指标非有限')
            if key in protocol['evaluations']:
                spec = protocol['evaluations'][key]
                if value.get('mask_spec', {}).get('rates') != [spec.get('rate',protocol['rate'])]:
                    issues.append('实测mask配置缺失率不符')
    failure = suite/'failures'/f'{name}.json'
    event = load(failure) if failure.exists() else None
    finished = bool(receipt) and evaluation.get('status') == 'finished' and set(evaluation.get('sets', {})) == set(protocol['evaluations']) and not issues
    diagnostics_nonfinite = sum(isinstance(v,str) and v.lower() in ('inf','-inf','nan')
        for r in history for split in ('train','val') for v in (r.get(split) or {}).values())
    record = dict(status='finished' if finished else 'failed' if issues or event else 'partial' if history else 'pending',
                  epoch=len(history), best_epoch=receipt.get('best_epoch'), val_mae=receipt.get('best_val_mae'),
                  minutes=receipt.get('total_time_sec',0)/60 if receipt else None,
                  parameters=meta.get('total_params'), trainable=meta.get('trainable_params'),
                  peak_gib=max((r.get('perf',{}).get('peak_memory_gb',0) for r in history), default=0),
                  amp_skips=sum(r['train'].get('train_skipped_amp_steps',0) for r in history),
                  nonfinite_diagnostics=diagnostics_nonfinite, issues=sorted(set(issues)), failure_event=event,
                  run=str(run) if run else None, rate=config['data']['mask']['missing_rate'],
                  evaluations=evaluation.get('sets',{}))
    return record, history


def export_report(root, suite, payload):
    root, suite = Path(root), Path(suite)
    manifest = suite/'reference_manifest.json'
    if manifest.exists():
        for rel, sha in load(manifest).items():
            if filehash(suite/rel) != sha:
                raise RuntimeError('Frozen reference changed: '+rel)
    definitions = dict(payload['references'])
    definitions.update({j['method']:j for j in payload['jobs'].values()})
    records = {}; histories = {}
    for label, job in definitions.items():
        records[label], histories[label] = collect(suite, label, job, label in payload['references'])
    final = all(r['status']=='finished' for r in records.values())
    date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    title = f'{date}_v24_历史池后续8组验证与4组探索_{suite.name}_{"完整分析" if final else "阶段分析"}'
    dest = root/'experments_report'/(title+'.md'); dest.parent.mkdir(parents=True, exist_ok=True)
    assets = dest.with_suffix(''); assets.mkdir(exist_ok=True)
    write(suite/'summary.json',records); write(assets/'results.json',records); write(assets/'curves.json',histories)
    lines = ['# '+title, '', '新增12组；同时披露旧X十组与K04参考。'+('全部完成。' if final else '阶段报告，尚未全部完成。'),
             '', '## 1. 具体方法与复现信息', '', '| 实验 | 来源 | 缺失率 | 具体做法 | 对照 |','|---|---|---|---|---|']
    for label, job in definitions.items():
        plan = job['config']['experiment_plan']
        lines.append(f'| {label} | {"复用" if label in payload["references"] else "新训练"} | {records[label]["rate"]} | {plan.get("method",job.get("method"))} | {plan.get("reference",job.get("reference","核心验证"))} |')
    doc = suite/'source_snapshot/scripts/v24/README_HISTORY_FOLLOWUP.md'
    if not doc.exists():
        doc = Path(__file__).with_name('README_HISTORY_FOLLOWUP.md')
    lines += ['', doc.read_text().split('## 方法与复现',1)[1].split('## 评估、报告与边界',1)[0],
              f'[冻结计划]({suite/"plan.json"})；[源码]({suite/"source_snapshot"})；[公共初始化]({suite/"initialization_audit.json"})；[GPU验收]({suite/"acceptance/preflight.json"})。',
              f'原X各组完整实现见[冻结说明]({Path(payload["upstream"])/"source_snapshot/scripts/v24/README_HISTORY_PLANE_INTERACTION.md"})。',
              '', '| 实验 | 配置 | 日志/检查点所在run |','|---|---|---|']
    for label, job in definitions.items():
        config = suite/'references'/label/'config.json' if label in payload['references'] else suite/'configs'/f'{job["variant"]}.json'
        lines.append(f'| {label} | [config]({config if config.exists() else suite/"plan.json"}) | {records[label]["run"] or "尚未启动"} |')
    lines += ['', '## 2. 完整结果、曲线与诊断', '', '| 组 | 状态/epoch | best | val MAE | ID MAE/RMSE | 分钟 | 峰值GiB | 总/可训参数 | ID ms/窗口 | AMP跳步/非有限诊断数 |', '|---|---|---|---|---|---|---|---|---|---|']
    for label, r in records.items():
        m = r['evaluations'].get('in_distribution',{}).get('metrics',{})
        lines.append(f'| {label} | {r["status"]}/{r["epoch"]} | {r["best_epoch"]} | {fmt(r["val_mae"])} | {fmt(m.get("mae"))}/{fmt(m.get("rmse"))} | {fmt(r["minutes"])} | {fmt(r["peak_gib"])} | {r["parameters"]}/{r["trainable"]} | {fmt(m.get("forward_ms_per_sample_per_rank"))} | {r["amp_skips"]}/{r["nonfinite_diagnostics"]} |')
        if r['issues'] or r['failure_event']:
            lines += ['', f'{label} 核验问题/历史失败：{r["issues"]}；{r["failure_event"]}', '']
    lines += ['', '分钟为epoch训练＋验证，不含六套评估/落盘；GiB是allocated峰值；推理计时为测试过程单次测量。非有限诊断数统计压缩epoch日志，原始详细诊断仍保留在run/logs中。',
              '', '| 组 | 协议 | MAE | RMSE |','|---|---|---|---|']
    for label, r in records.items():
        for protocol, e in r['evaluations'].items():
            lines.append(f'| {label} | {protocol} | {fmt(e["metrics"].get("mae"))} | {fmt(e["metrics"].get("rmse"))} |')
    lines += ['', '| 组 | 协议 | 缺失族 | MAE | RMSE |','|---|---|---|---|---|']
    for label, r in records.items():
        for protocol, e in r['evaluations'].items():
            for key, value in sorted(e['metrics'].items()):
                if key.startswith('coe_family_') and key.endswith('_mae'):
                    lines.append(f'| {label} | {protocol} | {key[11:-4]} | {fmt(value)} | {fmt(e["metrics"].get(key[:-3]+"rmse"))} |')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v24_y_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rate in (.4,.6):
        fig, axs = plt.subplots(2,2,figsize=(13,9))
        for label, history in histories.items():
            if records[label]['rate'] != rate:
                continue
            for ax, (split, metric) in zip(axs.flat,[('train','mae'),('val','mae'),('train','loss'),('val','rmse')]):
                points = [r for r in history if r.get(split)]
                if points:
                    ax.plot([r['epoch'] for r in points],[r[split][metric] for r in points],label=label)
                ax.set_title(f'rate {rate} {split} {metric}'); ax.set_xlabel('epoch'); ax.grid(alpha=.2)
        for ax in axs.flat:
            if ax.lines:
                ax.legend(fontsize=7,ncol=3)
        filename = f'curves_rate{rate}.png'; fig.tight_layout(); fig.savefig(assets/filename,dpi=140); plt.close(fig)
        lines += ['', f'![rate{rate}曲线]({assets.name}/{filename})']
    lines += ['', '有/无中间监督的总loss定义不同，不按loss高低排名。', '', '| 组 | train MAE 首→末 | val MAE 首→末 | best距训练结束epoch |', '|---|---|---|---|']
    for label, history in histories.items():
        vals = [r for r in history if r.get('val')]
        if history:
            valtext = f'{fmt(vals[0]["val"]["mae"])}→{fmt(vals[-1]["val"]["mae"])}' if vals else '尚无验证'
            gap = history[-1]['epoch']-records[label]['best_epoch'] if records[label]['best_epoch'] else '—'
            lines.append(f'| {label} | {fmt(history[0]["train"]["mae"])}→{fmt(history[-1]["train"]["mae"])} | {valtext} | {gap} |')
    lines += ['', '| 组 | 协议 | 轮次 | MAE | RMSE | 改善比例 | 退步比例 | 修好比例 | 破坏比例 |','|---|---|---|---|---|---|---|---|---|']
    for label, r in records.items():
        for protocol, e in r['evaluations'].items():
            for step in range(1,5):
                values = [fmt(e['metrics'].get(f'coe_x_step{step}_{k}')) for k in ('mae','rmse','improved_fraction','worsened_fraction','repaired_fraction','broken_fraction')]
                lines.append(f'| {label} | {protocol} | {step} | '+' | '.join(values)+' |')
    lines += ['', '修好/破坏阈值=0.05×max(观测RMS,1)，各比例以有效缺失点为分母；K04无新逐轮指标时留空。',
              '', '| 组 | 轮 | 历史P1/P2/P3权重 | 历史幅度 | 全局TH/TW/HW权重 | 全局平面幅度 | 全局平面执行次数/窗口 | 交互幅度 |','|---|---|---|---|---|---|---|---|']
    for label, r in records.items():
        m = r['evaluations'].get('in_distribution',{}).get('metrics',{})
        for step in range(1,5):
            value = lambda k:fmt(m.get(f'coe_x_step{step}_{k}'))
            lines.append(f'| {label} | {step} | '+ '/'.join(value(f'pool_weight_P{i}') for i in (1,2,3))+' | '+value('pool_delta_abs')+' | '+ '/'.join(value(f'global_plane_{v}_weight') for v in ('TH','TW','HW'))+' | '+value('global_plane_delta_abs')+' | '+value('global_plane_executions')+' | '+value('interaction_abs')+' |')
    lines += ['',f'[全部原始评估/路径/门/分支诊断]({assets.name}/results.json)；[全部训练曲线数据]({assets.name}/curves.json)。',
              '', '## 3. 配对分析、负结果、证据边界与建议', '', '相对变化=100×(候选/对照−1)，负数为改善。',
              '', '| 候选/对照 | 协议 | MAE变化 | RMSE变化 |','|---|---|---|---|']
    for a,b in PAIRS:
        if records[a]['status'] != 'finished' or records[b]['status'] != 'finished':
            continue
        changes = {}
        for key, e in records[a]['evaluations'].items():
            ref = records[b]['evaluations'][key]['metrics']
            d = [100*(e['metrics'][m]/ref[m]-1) for m in ('mae','rmse')]; changes[key]=d[0]
            lines.append(f'| {a}/{b} | {key} | {d[0]:+.2f}% | {d[1]:+.2f}%'+(' **退步>5%**' if max(d)>5 else '')+' |')
        if all(k in changes for k in PRIMARY):
            lines += ['',f'{a}/{b}：ID MAE {changes["in_distribution"]:+.2f}%；三类主要OOD平均相对变化 {sum(changes[k] for k in PRIMARY)/3:+.2f}%，最差 {max(changes[k] for k in PRIMARY):+.2f}%。两套mask复测单列。','']
    lines += ['', '| 组 | 同率纯CMFF参考 | ID MAE变化 | 三类OOD平均MAE变化 | 最差OOD变化 |','|---|---|---|---|---|']
    for label,r in records.items():
        ref = records['Y05' if r['rate']==.6 else 'X01']; keys=('in_distribution',*PRIMARY)
        if r['status']=='finished' and ref['status']=='finished' and all(k in r['evaluations'] and k in ref['evaluations'] for k in keys):
            d={k:100*(r['evaluations'][k]['metrics']['mae']/ref['evaluations'][k]['metrics']['mae']-1) for k in keys}
            lines.append(f'| {label} | {"Y05" if r["rate"]==.6 else "X01"} | {d["in_distribution"]:+.2f}% | {sum(d[k] for k in PRIMARY)/3:+.2f}% | {max(d[k] for k in PRIMARY):+.2f}% |')
    lines += ['', '| 四格(组合,单项A,单项B,基线) | 协议 | MAE差分之差 |','|---|---|---|']
    for cells in [('X05','K04','X06','X01'),('Y04','Y02','Y03','Y01'),('Y08','Y06','Y07','Y05'),('Y11','Y10','X05','K04'),('Y12','X10','X05','K04')]:
        if all(records[n]['status']=='finished' for n in cells):
            for key in ('in_distribution',*PRIMARY):
                v=[records[n]['evaluations'][key]['metrics']['mae'] for n in cells]
                lines.append(f'| {"/".join(cells)} | {key} | {v[0]-v[1]-v[2]+v[3]:+.4f} |')
    lines += ['', '差分之差为原始误差组合−单项A−单项B＋基线，负值表示在该单seed上超过原始误差加性预期；不是显著性检验或已证明协同机制。',
              '解释顺序：Y01–Y04检验同率中间监督，并在有监督下分离差分/历史池；Y05–Y08检验0.6训练下四格对照，不能称为0.4到0.6迁移。Y09/Y10匹配参数仅换融合权重；与X07/X08相比同时改变位置、维度与算力，不能只归因于全专家可见。Y11/Y12需要同时看两个单项参考及组合四格，避免把单项收益重算成组合创新。',
              '中间监督是常规深监督工具，不作为新贡献。中间轮误差下降需实测，不保证逐轮单调；历史权重与平面覆盖不是校准置信度。固定尺度调度与动态专家选择必须分开表述。',
              '本批所有结论仅限TaxiBJ seed7；已见测试用于探索，mask复测不是训练seed复验。完整披露ID/OOD退步和成本；未设事后ID退步阈值，不按测试选择checkpoint。',
              '下一步：以同率直接对照、逐轮预测质量和成本共同筛选方向，用独立训练种子/数据集确认；全专家平面若有收益，补等容量及执行成本对照。未完成时不根据阶段结果改变冻结训练计划。']
    dest.write_text('\n'.join(lines)+'\n'); (suite/'report_path.txt').write_text(str(dest)+'\n')
    return dest
