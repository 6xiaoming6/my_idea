#!/usr/bin/env python3
"""Report all eleven budgets plus the read-only I37 100-epoch reference."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from release_utils import load, write
from epoch_budget_common import completed, scores

LABELS = ['ID', '二元组合', '未见形态', '三元组合', '二元复测', '形态复测']


def report(suite):
    suite = Path(suite); plan = load(suite / 'plan.json'); source = Path(plan['source'])
    protocol = load(source / 'configs/test_protocol.json')
    refdir = Path(plan['reference']); ref = load(refdir / 'evaluation.json')
    refid = ref['sets']['in_distribution']['metrics']['mae']
    items = []
    for job in plan['jobs']:
        run = suite / 'runs' / job['name']
        result = load(run / 'result.json') if (run / 'result.json').exists() else {}
        ev = load(run / 'evaluation.json') if (run / 'evaluation.json').exists() else {}
        history = [json.loads(v) for v in (run / 'logs/metrics.jsonl').read_text().splitlines()] if (run / 'logs/metrics.jsonl').exists() else []
        try:
            done = completed(run, job, load(source / job['config']), protocol)
            error = ''
        except (AssertionError, KeyError) as exc:
            done = False; error = repr(exc)
        items.append(dict(**job, run=run, result=result, ev=ev, history=history, done=done, error=error))
    all_done = all(v['done'] for v in items)
    date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    target = Path(plan['report_dir']) / f'{date}_best_I37训练轮数与最低学习率对照_分析报告.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    status = '完整报告' if all_done else '阶段报告（存在未完成或失败组）'
    lines = [f'# I37训练轮数与最低学习率对照：{status}', '',
             f'日期：{date}；模型基准 `{plan["base_commit"]}`，实验冻结提交 `{plan["source_commit"]}`；训练seed7。', '',
             '## 一、实验具体做法与复现配置', '',
             '| 编号 | 总轮数 | 最低学习率 | 对照与问题 |', '| --- | ---: | ---: | --- |',
             '| I37参考 | 100 | 3e-4 | 已完成发布版，读取已有结果，不重复训练 |']
    for job in items:
        lines.append(f'| {job["name"]} | {job["epochs"]} | {job["eta_min"]:g} | 同最低学习率比较训练预算，同轮数比较最低学习率 |')
    lines += ['', '### 模型与训练协议', '',
              '保持I37原版[B,64,12,32,32]隐藏状态，真实F/M/C观测融合H0，四轮共享八专家，每轮原生Top-2，前三轮自由尺度、末轮F；保留受控差分、历史池、原Decoder和学习分配投影。没有加入latent压缩、空间Linear或可学习损失权重。', '',
              'TaxiBJ双通道、12时间步，F/M/C为32/16/8，sum聚合，各尺度独立mask；沿用四类混合mask0.4和原train/val/test划分。seed7，batch32，workers0，GPU1，CPU线程2，AMP、AdamW、weight_decay=1e-4、梯度裁剪1。每5轮验证，无早停。', '',
              '所有新增组从相同未训练初始化独立训练，初始学习率1e-3。每组的余弦总周期与本组训练预算一致，末轮训练学习率到达指定eta_min。120–200轮不是从原100轮last继续训练；不同预算的前100轮学习率轨迹也不同。这是在比较完整训练计划，不是单独隔离新增epoch数量的效果。', '',
              '投影前/后F缺失L1权重固定0.5/0.5，可见F重建0.1，可见M/C投影前一致性0.1，候选均衡0.01。仍仅按ID验证MAE选best；每组best测试原六套协议，不按OOD或测试集挑训练检查点。', '',
              'H后缀表示eta_min=3e-4，L表示1e-5。100H读取I37参考；队列先100L，再120H/120L、140H/140L，直至200H/200L。所有src及原train/evaluate入口与best基准逐文件校验未变。训练run中variant沿用I37_seed7，具体预算以目录、冻结配置及plan为准。', '',
              f'源码：`{source}`；配置：`configs/epoch_budget/<编号>.json`；运行目录：`{suite}/runs/<编号>`；[冻结计划]({suite / "plan.json"})；源码、配置、数据、初始化及协议摘要见`manifest.json`。', '',
              f'恢复入口：`{plan["python"]} {source}/scripts/queue_epoch_budget.py --suite {suite}`。完成组跳过，中断组从完整last恢复；失败停止，不自动重试。每run保留best/last和15秒GPU日志；最终自动导出本报告并结束专用tmux会话。', '',
              '## 二、完整结果与诊断', '',
              '| 实验 | 完成轮数/预算 | best轮次 | 验证MAE | 训练验证分钟 | 峰值allocated GiB | 状态 |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for item in items:
        h = item['history']; r = item['result']
        epoch = r.get('completed_epochs', h[-1]['epoch'] if h else 0)
        peak = max((x.get('perf', {}).get('peak_memory_gb', 0) for x in h), default=0)
        lines.append(f'| {item["name"]} | {epoch}/{item["epochs"]} | {r.get("best_epoch", "—")} | {r.get("best_val_mae", "—")} | {r.get("total_time_sec", 0)/60:.2f} | {peak:.3f} | {"完成" if item["done"] else "未完成/失败 " + item["error"]} |')
    lines += ['', '耗时为原训练入口记录的训练与验证累计，不含全部I/O和六套测试；显存为PyTorch allocated，并非nvidia-smi进程占用。失败、未完成或异常组不纳入排名，仍完整列出。', '',
              '| 实验 | 测试协议 | MAE | RMSE |', '| --- | --- | ---: | ---: |']
    evaluations = [('I37', ref)] + [(v['name'], v['ev']) for v in items]
    for name, ev in evaluations:
        for key, label in zip(protocol['evaluations'], LABELS):
            m = ev.get('sets', {}).get(key, {}).get('metrics', {})
            lines.append(f'| {name} | {label} | {m.get("mae", "—")} | {m.get("rmse", "—")} |')
    lines += ['', '| 实验 | ID MAE变化% | OOD MAE均值 | OOD类别比值均值 | OOD RMSE均值 | 最差类别MAE变化% |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    comparisons = {'I37': scores(ref, ref)}; eligible = []
    for item in items:
        if not item['done']: continue
        ev = item['ev']; s = scores(ev, ref); comparisons[item['name']] = s
        ident = ev['sets']['in_distribution']['metrics']['mae']
        lines.append(f'| {item["name"]} | {100*(ident/refid-1):+.3f} | {s["mae"]["mean"]:.6f} | {s["mae"]["ratio"]:.6f} | {s["rmse"]["mean"]:.6f} | {s["mae"]["worst_pct"]:+.3f} |')
        if ident <= refid * 1.03:
            eligible.append((s['mae']['ratio'], s['rmse']['ratio'], ident, item['name']))
    lines += ['', 'OOD先合并二元组合的两次mask测试、未见形态的两次mask测试，再对二元、形态、三元三类别等权平均。比值按类别分别相对I37计算后平均。沿用ID代价≤3%的探索筛选线，完整展示未达门槛者。', '',
              '### 训练曲线与预算末期趋势', '',
              '| 实验 | 首轮loss | 末轮loss | 第100轮验证MAE | 末轮验证MAE | best后继续训练轮数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in items:
        h = item['history']
        if not h: continue
        at100 = next((r.get('val') for r in h if r['epoch'] == 100), None) or {}
        lastval = next((r['val'] for r in reversed(h) if r.get('val')), {})
        best = item['result'].get('best_epoch')
        lines.append(f'| {item["name"]} | {h[0]["train"]["loss"]:.4f} | {h[-1]["train"]["loss"]:.4f} | {at100.get("mae", "—")} | {lastval.get("mae", "—")} | {h[-1]["epoch"]-best if best else "—"} |')
    os.environ.setdefault('MPLCONFIGDIR', str(suite / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets = target.parent / (target.stem + '_assets'); assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    refhistory = [json.loads(v) for v in (refdir / 'logs/metrics.jsonl').read_text().splitlines()] if (refdir / 'logs/metrics.jsonl').exists() else []
    for name, h in [('I37', refhistory)] + [(v['name'], v['history']) for v in items]:
        if not h: continue
        axes[0].plot([r['epoch'] for r in h], [r['train']['loss'] for r in h], label=name)
        val = [r for r in h if r.get('val')]
        axes[1].plot([r['epoch'] for r in val], [r['val']['mae'] for r in val], label=name)
        axes[2].plot([r['epoch'] for r in h], [r['train'].get('lr', float('nan')) for r in h], label=name)
    for ax, title in zip(axes, ['Train loss', 'ID validation MAE', 'Training learning rate']):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=6, ncol=2)
    fig.savefig(assets / 'curves.png', dpi=170); fig.savefig(assets / 'curves.pdf'); plt.close(fig)
    lines += ['', f'![训练、验证与学习率曲线]({assets.name}/curves.png)', '',
              '| 实验 | 投影前F L1 | 投影后F L1 | 无约束F MAE | 有效计数 | 主尺度路径 |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for name, ev in evaluations:
        m = ev.get('sets', {}).get('in_distribution', {}).get('metrics', {})
        paths = sorted(((k, v) for k, v in m.items() if k.startswith('coe_scale_execution_path_') and k.endswith('_fraction') and isinstance(v, (int, float))), key=lambda x: -x[1])[:2]
        text = ', '.join(k.removeprefix('coe_scale_execution_path_').removesuffix('_fraction') + f':{v:.3f}' for k, v in paths)
        count = m.get('combination_post_unconstrained_count', 0)
        lines.append(f'| {name} | {m.get("l_raw_f", "—")} | {m.get("l_final", "—")} | {m.get("combination_post_unconstrained_mae", "—") if count else "—"} | {count} | {text or "—"} |')
    lines += ['', 'F损失诊断采用损失日志的归约口径，与全局主指标MAE不完全相同；不能混用。空区域误差显示“—”。详细逐轮通信、历史和路由诊断保留于各run的evaluation.json及logs/diagnostics。', '',
              '## 三、分析、证据边界与建议', '',
              '### 同轮数最低学习率对照', '',
              '| 轮数 | 高最低学习率参考 | 低最低学习率组 | ID MAE变化% | OOD MAE均值变化% |',
              '| --- | --- | --- | ---: | ---: |']
    evdict = dict(evaluations)
    for n in (100, 120, 140, 160, 180, 200):
        high = 'I37' if n == 100 else f'E{n}_H'; low = f'E{n}_L'
        if high not in comparisons or low not in comparisons: continue
        x = evdict[high]['sets']['in_distribution']['metrics']['mae']; y = evdict[low]['sets']['in_distribution']['metrics']['mae']
        lines.append(f'| {n} | {high} | {low} | {100*(y/x-1):+.3f} | {100*(comparisons[low]["mae"]["mean"]/comparisons[high]["mae"]["mean"]-1):+.3f} |')
    lines += ['', '### 同最低学习率预算对照', '',
              '| 最低学习率 | 相邻预算 | ID MAE变化% | OOD MAE均值变化% |', '| --- | --- | ---: | ---: |']
    for label, eta in [('H', 3e-4), ('L', 1e-5)]:
        for n in (120, 140, 160, 180, 200):
            a = 'I37' if label == 'H' and n == 120 else f'E{n-20}_{label}'; b = f'E{n}_{label}'
            if a not in comparisons or b not in comparisons: continue
            x = evdict[a]['sets']['in_distribution']['metrics']['mae']; y = evdict[b]['sets']['in_distribution']['metrics']['mae']
            lines.append(f'| {eta:g} | {a}→{b} | {100*(y/x-1):+.3f} | {100*(comparisons[b]["mae"]["mean"]/comparisons[a]["mae"]["mean"]-1):+.3f} |')
    if eligible:
        winner = sorted(eligible)[0]
        lines += ['', f'ID代价≤3%的新组中，OOD类别比值最低为{winner[3]}：{winner[0]:.6f}。' + ('仍未超过I37，不强行推荐新组。' if winner[0] >= 1 else '本次seed7汇总优于I37，仍须结合最差类别、RMSE和成本判断。')]
    else:
        lines += ['', '当前没有已完成新组满足ID代价≤3%门槛，不能据不完整结果推荐替代I37。']
    lines += ['', '实验同时改变优化步数及余弦衰减速度，不能把差别全归因于多训练若干轮。更低eta_min也不保证更好的OOD；要核对验证曲线是否改善、best是否提前出现以及各类OOD是否一致。所有组仍为seed7，mask复测不等于独立训练种子，测试结果上的超参数探索不能宣称独立确认或统计显著性。', '',
              '本报告不自动更改best分支或覆盖发布模型。如果训练预算收益有限，保留原100轮；若存在更好的折中，再依据完整表格决定训练设置。', '']
    target.write_text('\n'.join(lines))
    write(suite / 'comparison.json', comparisons)
    write(suite / 'report_receipt.json', dict(path=str(target), complete=all_done))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--suite', type=Path, required=True)
    print(report(parser.parse_args().suite))
