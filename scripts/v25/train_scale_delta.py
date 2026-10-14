#!/usr/bin/env python3
"""Unmodified epoch trainer with crash traces, without a periodic stack walker."""
import argparse,faulthandler,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(add_help=False);p.add_argument('--run-dir',required=True)
a,_=p.parse_known_args();log=Path(a.run_dir)/'logs/thread_stacks.log';log.parent.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
with log.open('a') as f:
    faulthandler.enable(file=f,all_threads=True)
    try:runpy.run_path(str(ROOT/'scripts/v24/train_four_direction.py'),run_name='__main__')
    finally:
        faulthandler.disable()
