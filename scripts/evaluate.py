#!/usr/bin/env python3
"""Evaluate a saved I37 best checkpoint on the frozen six test protocols."""
import argparse
import os
from pathlib import Path
from release_utils import load
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.device == 'cuda': os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    from evaluate_core import evaluate_sets
    import torch
    torch.set_num_threads(a.threads)
    from summarize import summarize
    evaluate_sets(a.checkpoint, load(ROOT / 'configs/test_protocol.json'), a.data_dir / 'taxibj_test.npz',
                  a.output, device=a.device)
    summarize(a.output, a.output.with_suffix('.md'))


if __name__ == '__main__': main()
