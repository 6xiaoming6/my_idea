"""Formal, source-linked paired report; partial runs are explicitly marked."""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path
from run_b3_c3 import load, write, digest


def fmt(x):
    return f'{x:.4f}' if isinstance(x, (int, float)) and math.isfinite(x) else '—'


def export_report(root, suite, jobs):
    root, suite = Path(root), Path(suite)
    rows, curves, diagnostics = [], {}, {}
    for n, j in jobs.items():
        rp = suite / 'results' / f'{n}.json'
        ep = suite / f'evaluations_rate{j["rate"]:g}' / f'{n}.json'
        r, e = (load(rp) if rp.exists() else {}), (load(ep) if ep.exists() else {})
        expected = j['config']['train']['epochs']
        pointer = suite / 'runs' / f'{n}.json'
        run = Path(r['run_dir']) if r else Path(load(pointer)['run_dir']) if pointer.exists() else None
        hp = run / 'logs/metrics.jsonl' if run else None
        hist = [json.loads(x) for x in hp.read_text().splitlines()] if hp and hp.exists() else []
        issues = []
        if r:
            if r.get('completed_epochs') != expected or r.get('config_sha256') != digest(j['config']):
                issues.append('training receipt mismatch')
            for cp in ('best.pth', 'last.pth'):
                if not (run / 'checkpoints' / cp).exists():
                    issues.append('missing ' + cp)
            if len(hist) != expected:
                issues.append('incomplete metric history')
            if digest(load(run / 'config.json')) != digest(j['config']):
                issues.append('saved config mismatch')
        if e and (e.get('config_sha256') != digest(j['config']) or e.get('protocol_sha256') != digest(j['config']['experiment_plan']['protocol'])):
            issues.append('evaluation receipt mismatch')
        for h in hist:
            for split in ('train', 'val'):
                if h.get(split) and any(not isinstance(h[split].get(k), (int, float)) or not math.isfinite(h[split][k]) for k in ('loss', 'mae', 'rmse')):
                    issues.append(f'nonfinite primary epoch {h["epoch"]} {split}')
        sets = e.get('sets', {})
        for key, val in sets.items():
            if any(not isinstance(val['metrics'].get(k), (int, float)) or not math.isfinite(val['metrics'][k]) for k in ('mae', 'rmse')):
                issues.append('nonfinite evaluation ' + key)
        valid_evaluation = e.get('status') == 'finished' and set(sets) == set(j['config']['experiment_plan']['protocol']['evaluations'])
        status = 'finished' if r.get('status') == 'finished' and valid_evaluation and not issues else 'trained' if r else 'partial' if hist else 'pending'
        failure = suite / 'failures' / f'{n}.json'
        if failure.exists() and status != 'finished':
            status = 'failed'; issues.append(load(failure)['error'])
        meta = load(run / 'training_metadata.json') if run and (run / 'training_metadata.json').exists() else {}
        row = {'variant': n, 'dataset': j['dataset'], 'rate': j['rate'], 'method': j['method'], 'status': status,
               'completed_epochs': len(hist), 'best_epoch': r.get('best_epoch'), 'val_mae': r.get('best_val_mae'),
               'run_dir': str(run) if run else None, 'training_seconds': r.get('total_time_sec'),
               'params': meta.get('total_params'), 'trainable_params': meta.get('trainable_params'),
               'peak_gib': max((h.get('perf', {}).get('peak_memory_gb', 0) for h in hist), default=0),
               'issues': issues, 'evaluations': {k: {m: v['metrics'][m] for m in ('mae', 'rmse')} for k, v in sets.items()}}
        rows.append(row); curves[n] = hist
        diagnostics[n] = sets
    write(suite / 'summary.json', rows)
    completed = sum(r['status'] == 'finished' for r in rows)
    final = completed == len(rows)
    report_dir = root / 'experments_report'; report_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    dest = report_dir / f'{date}_v24_双数据集四缺失率尺度对比_{suite.name}_{"完整分析" if final else "阶段分析"}.md'
    assets = dest.with_suffix(''); assets.mkdir(exist_ok=True)
    write(assets / 'metrics.json', rows)
    write(assets / 'curves.json', curves)
    write(assets / 'evaluation_diagnostics.json', diagnostics)
    lines = [f'# {date} v24 双数据集四缺失率：固定CMFF与自由三尺度Top-1', '',
             f'训练及评估完成：{completed}/{len(rows)}。' + ('完整批次。' if final else '阶段报告，不能当作最终结论。'), '',
             '## 1. 各组具体做法与复现配置', '',
             '| 组 | 数据集 | random缺失率 | 具体方法 | 对照与问题 |', '| --- | --- | --- | --- | --- |']
    for n, j in jobs.items():
        method = '固定C→M→F→F' if j['method'] == 'cmff' else '三尺度softmax预热5epoch＋过渡5epoch→四轮自由Top-1，81路径'
        lines.append(f'| {n} | {j["dataset"]} | {j["rate"]} | {method} | {j["reference"]}：自由尺度是否优于固定尺度 |')
    c = next(iter(jobs.values()))['config']; u = c['model']['coe']['id_priority']; t = c['train']
    lines += ['', f'共同条件：seed{c["seed"]}，batch{c["data"]["batch_size"]}，{t["epochs"]}epoch，每{t["val_epoch"]}epoch验证；AdamW，1e-3余弦至3e-4，weight_decay=1e-4，clip=1，AMP，无早停。GPU单卡串行。',
              f'四轮共享八专家T/S/TD/SD/TA/ST/TL/SL、原生Top2和组内softmax、direct更新、关闭completion feedback。通信={u["communication"]}；delta时采用±0.1 RMS匹配状态差分。L1+0.01 candidate balance。无额外视图、尺度辅助损失；专家路由本身不预热。',
              '尺度为空间F=原网格、M=1/2、C=1/4；T保持12。TaxiBJ为32×32/16×16/8×8，BikeNYC为24×12/12×6/6×3。每个执行尺度只激活同一原生Router选出的两位专家，粗输出上采样后传递。固定模型及Top1推理每窗口8次专家执行。面积代理不是FLOPs。',
              '自由组训练epoch1–5执行三个尺度并按softmax融合；epoch6–10权重为a*p+(1-a)*(hard+p-stopgrad(p))，a=5/6,4/6,3/6,2/6,1/6；epoch11起a=0且仅执行选中尺度。验证/测试始终Top1。预热最多24次专家执行/窗口，额外成本单列；尺度历史计数按融合贡献累计，实际执行量独立记录。100epoch不额外延长。',
              '自由尺度头读取当前路由特征、F/M/C累计次数比例和剩余轮数比例，零输出层+CMFF微偏置初始化，选中分支直通梯度训练；推理argmax。固定方法拥有相同但冻结的尺度头，总参数相同、可训练参数不同。',
              'random沿用四种基础缺失混合：random_point/node_outage/temporal_gap/spatial_region，近似均衡；训练逐epoch重采样，验证测试固定。基础mask种子20260917，val/test偏移20000/30000。每个缺失率独立从头训练，训练、验证与ID测试率相同；不是0.4模型的缺失率迁移。',
              '开启严格确定性、禁用TF32，空间池化和插值使用确定性实现。每数据集固定同seed公共初始化，两种尺度策略初始状态字典完全相同。best只依ID验证MAE选择；保存best和完整恢复last。六套评估率均与训练率一致。',
              f'冻结计划：[plan]({suite / "plan.json"})；冻结源码：[source]({suite / "source_snapshot"})；初始化核验：[audit]({suite / "initialization_audit.json"})。',
              f'恢复：`python -u scripts/v24/run_scale_rate_compare.py --suite {suite} --gpu 0`。', '',
              '| 组 | 配置 | 原始日志 |', '| --- | --- | --- |']
    for r in rows:
        n = r['variant']; log = str(Path(r['run_dir']) / 'logs') if r['run_dir'] else ''
        lines.append(f'| {n} | [config]({suite / "configs" / (n+".json")}) | ' + (f'[logs]({log})' if log else '尚未启动') + ' |')
    lines += ['', '## 2. 完整结果、训练曲线与诊断', '',
              '| 组 | 状态/epoch | best epoch | val MAE | ID MAE | ID RMSE | 训练分钟 | 峰值GiB | 总/可训参数 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        v = r['evaluations'].get('in_distribution', {})
        lines.append(f'| {r["variant"]} | {r["status"]}/{r["completed_epochs"]} | {r["best_epoch"]} | {fmt(r["val_mae"])} | {fmt(v.get("mae"))} | {fmt(v.get("rmse"))} | {fmt(r["training_seconds"]/60 if r["training_seconds"] else None)} | {fmt(r["peak_gib"])} | {r["params"]}/{r["trainable_params"]} |')
        if r['issues']:
            lines.append('\n核验异常：' + r['variant'] + ': ' + '; '.join(r['issues']) + '\n')
    lines += ['', '| 组 | 协议 | MAE | RMSE |', '| --- | --- | --- | --- |']
    for r in rows:
        for name, m in r['evaluations'].items():
            lines.append(f'| {r["variant"]} | {name} | {fmt(m["mae"])} | {fmt(m["rmse"])} |')
    lines += ['', '| 组 | 专家调用/窗口 | 面积代理 | ID最大尺度路径及占比 |', '| --- | --- | --- | --- |']
    for n, sets in diagnostics.items():
        m = sets.get('in_distribution', {}).get('metrics', {})
        path = {k: v for k, v in m.items() if k.startswith('coe_scale_path_') and isinstance(v, (int, float))}
        largest = max(path, key=path.get) if path else None
        lines.append(f'| {n} | {fmt(m.get("coe_expert_execution_count"))} | {fmt(m.get("coe_expert_grid_equivalents"))} | {largest or "详见诊断"}: {fmt(path.get(largest))} |')
    # Preserve complete protocol/family/route diagnostics separately; no giant CSV/log dump.
    lines += ['', f'[完整逐缺失族和路径诊断]({assets.name}/evaluation_diagnostics.json)；[全部训练曲线数据]({assets.name}/curves.json)。', '']
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/v24_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in dict.fromkeys(j['dataset'] for j in jobs.values()):
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), squeeze=False)
        for idx, rate in enumerate(dict.fromkeys(j['rate'] for j in jobs.values() if j['dataset'] == dataset)):
            for n, j in jobs.items():
                if (j['dataset'], j['rate']) != (dataset, rate):
                    continue
                hist = curves[n]
                for ax, split in ((axes[0, idx], 'train'), (axes[1, idx], 'val')):
                    points = [h for h in hist if h.get(split)]
                    ax.plot([h['epoch'] for h in points], [h[split]['mae'] for h in points], label=j['method'])
                    ax.set_title(f'{dataset} rate={rate} {split}'); ax.set_xlabel('epoch'); ax.set_ylabel('MAE')
                    if points: ax.legend()
        fig.tight_layout(); path = assets / f'{dataset}_curves.png'; fig.savefig(path, dpi=130); plt.close(fig)
        lines.append(f'![{dataset} curves]({assets.name}/{path.name})')
    lines += ['', '## 3. 配对分析、证据边界与建议', '',
              '| 数据集 | 率 | Top1相对CMFF ID MAE变化 | ID RMSE变化 |', '| --- | --- | --- | --- |']
    lookup = {r['variant']: r for r in rows}; changes = {}
    for n, j in jobs.items():
        if j['method'] != 'top1': continue
        if j['reference'] not in lookup:
            lines.append(f'| {j["dataset"]} | {j["rate"]} | 缺少CMFF对照 | — |')
            continue
        a, b = lookup[n], lookup[j['reference']]
        if a['status'] == b['status'] == 'finished':
            am, bm = a['evaluations']['in_distribution'], b['evaluations']['in_distribution']
            delta = 100 * (am['mae'] / bm['mae'] - 1)
            changes.setdefault(j['dataset'], []).append(delta)
            lines.append(f'| {j["dataset"]} | {j["rate"]} | {delta:+.2f}% | {100*(am["rmse"]/bm["rmse"]-1):+.2f}% |')
        else:
            lines.append(f'| {j["dataset"]} | {j["rate"]} | 未完成配对 | — |')
    for dataset, values in changes.items():
        lines.append(f'\n{dataset}：已完成{len(values)}个配对，Top1在{sum(v<0 for v in values)}个率上ID MAE更低；配对相对变化均值{sum(values)/len(values):+.2f}%。负数为改善。')
    lines += ['', '分别判断每个数据集、每个缺失率，不把两个数据集的原始MAE直接平均。不因路由更分散就认定有效，也不把8次专家激活视为相同FLOPs；自由尺度的实际面积可能更多或更少。',
              '本批每组只有一个训练种子；测试mask复测不等于训练种子复验。不同数据集均独立训练，不能称为跨数据集零样本迁移。每个缺失率重训的结果不能与同一模型跨缺失率迁移混淆。',
              '若自由尺度在多个率上获益，下一步仅对这些预先明确的配对补充训练种子；若只在部分率获益，定位适用范围，并结合实际路径/成本分析。无稳定收益时保留固定CMFF。',
              '当前尚未整体完成，所有比较均为阶段性结果。' if not final else '全部配对完成；单种子结果仍需复验后才能判断稳定优势。']
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (suite / 'report_path.txt').write_text(str(dest) + '\n')
    return dest
