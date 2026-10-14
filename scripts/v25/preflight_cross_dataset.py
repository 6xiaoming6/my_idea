#!/usr/bin/env python3
"""Reuse all native/heavy real-batch AMP checks on physical GPU1."""
import argparse,os
from pathlib import Path
import preflight
from run_cross_dataset_validation import gpu1_only

def run(suite):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Physical GPU1 required')
    # preflight checks the visibility string before opening CUDA; remove only its
    # original GPU0 admission guard via an explicit parameter (no GPU remapping).
    preflight.run(suite,gpu=1,check_gpu=gpu1_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();run(a.suite)
