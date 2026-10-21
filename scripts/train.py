#!/usr/bin/env python3
"""Train the frozen I37 method; resume automatically from a complete last.pth."""
import argparse
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
from release_utils import load, write, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/i37_taxibj.json')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--gpu', type=int, default=1, help='Physical GPU index, CUDA mode only; defaults to GPU1')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--evaluate', action='store_true', help='Run all six test protocols after training')
    p.add_argument('--stop-after', type=int, help='Epoch-boundary stop for continuation testing')
    a = p.parse_args()
    if a.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['MKL_NUM_THREADS'] = str(a.threads)
    from train_core import train
    import torch
    torch.set_num_threads(a.threads)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    cfg = load(a.config)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two invocations from sharing checkpoints/logs.
    import fcntl
    with (a.run_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {s: str((a.data_dir / f'taxibj_{s}.npz').resolve()) for s in ('train', 'val', 'test')}
        data_hashes = {s: sha256(path) for s, path in sources.items()}
        code_hashes = {str(f.relative_to(ROOT)): sha256(f) for folder in ('src', 'scripts')
                       for f in sorted((ROOT / folder).rglob('*.py'))}
        provenance = {'data_sha256': data_hashes, 'code_sha256': code_hashes}
        frozen = a.run_dir / 'run_provenance.json'
        if frozen.exists() and load(frozen) != provenance:
            raise ValueError('Run source/data changed; restore the original files before resuming')
        write(frozen, provenance)
        job = {'variant': f'I37_seed{cfg["seed"]}', 'config': cfg, 'sources': sources}
        write(a.run_dir / 'job.json', job)
        result = train(job, a.run_dir, a.run_dir / 'result.json', a.device, a.stop_after)
        print(result['status'], flush=True)
        if result['status'] == 'finished' and a.evaluate:
            from evaluate_core import evaluate_sets
            from summarize import summarize
            output = a.run_dir / 'evaluation.json'
            evaluate_sets(a.run_dir / 'checkpoints/best.pth', load(ROOT / 'configs/test_protocol.json'),
                          sources['test'], output, device=a.device)
            summarize(output, a.run_dir / 'summary.md')


if __name__ == '__main__':
    main()
