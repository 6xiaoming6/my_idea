#!/usr/bin/env python3
"""Evaluate a deliberately paused run using its immutable source and ID best."""
import argparse, fcntl, json, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--variant', required=True)
    a = p.parse_args(); suite = a.suite.resolve()
    snap = suite / 'source_snapshot'
    sys.path[:0] = [str(snap/'src'), str(snap/'scripts/v24'), str(snap/'scripts/v25')]
    import torch
    from run_b3_c3 import load, write, digest
    from prepare_data import sha
    from run_first_validation import verify
    from evaluate_four_direction import evaluate_sets
    from stmoe_imputer.utils.gpu_monitor import GPUMonitor
    from report_taxibj_input_communication import export_report
    torch.set_num_threads(2)
    plan = load(suite/'plan.json'); verify(suite, plan)
    job = load(suite/'resolved_jobs.json')[a.variant]
    run = Path(load(suite/'runs'/f'{a.variant}.json')['run_dir'])
    cp = run/'checkpoints/best.pth'
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    last = torch.load(run/'checkpoints/last.pth', map_location='cpu', weights_only=False)
    pause = load(suite/'pause_requested.json')
    if digest(saved['config']) != digest(job['config']):
        raise RuntimeError('Best checkpoint config mismatch')
    stamp = sha(cp)
    if pause['best_checkpoint_sha256'] != stamp:
        raise RuntimeError('Best checkpoint changed since pause')
    meta = {'status':'paused_by_user', 'time':time.time(), 'completed_epochs':last['epoch'],
            'best_epoch':saved['epoch'], 'best_val_mae':saved['metrics']['mae'],
            'checkpoint':str(cp), 'checkpoint_sha256':stamp,
            'planned_epochs':job['config']['train']['epochs']}
    del last, saved
    lock = Path(plan['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock'
    with lock.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = suite/'evaluations'/f'{a.variant}.json'
        with GPUMonitor(run):
            result = evaluate_sets(cp, job['config']['experiment_plan']['protocol'],
                                   job['sources']['test'], out, digest(job['config']))
        if sha(cp) != stamp:
            raise RuntimeError('Checkpoint changed during evaluation')
        meta.update(evaluation_status=result['status'], evaluation=str(out))
        write(suite/'paused_evaluation.json', meta)
        write(suite/'state.json', {'phase':'paused', **meta})
        report = export_report(Path(plan['project_root']), suite, load(suite/'resolved_jobs.json'))
        # Frozen report code predates intentional pauses. Add the explicit status
        # without fabricating a completed 100-epoch training receipt.
        text = report.read_text()
        marker = '\n## 1. '
        note = (f"\n用户主动暂停：已完成 {meta['completed_epochs']}/100 epoch；"
                f"ID 验证最佳为 epoch {meta['best_epoch']}，MAE {meta['best_val_mae']:.6f}。"
                "本报告测试均来自该 best.pth；六套评估已完成，训练未完成。"
                "队列中的 exit 143 为用户主动暂停产生，不能解释成训练崩溃；原始退出记录保留。"
                "与完整100轮实验的比较有训练预算差异。\n")
        text = text.replace(marker, note + marker, 1)
        report.write_text(text)
        print(json.dumps(meta, ensure_ascii=False), flush=True)
        for name, row in result['sets'].items():
            print(name, {k:row['metrics'][k] for k in ('mae','rmse')}, flush=True)
        print('Report:', report, flush=True)


if __name__ == '__main__':
    main()
