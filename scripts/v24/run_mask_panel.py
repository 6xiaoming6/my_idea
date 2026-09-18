#!/usr/bin/env python3
"""Evaluate existing best checkpoints on the same nine fixed mask families.

This is diagnostic only: it never chooses checkpoints and runs one model at a
time. It does not launch training. The default suite is the completed ABCDE
suite; pass --suite to use a different immutable suite.
"""
from __future__ import annotations
import argparse,csv,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,default=ROOT/'outputs/v24-COE/experiments/abcde/abcde/38027eefadf2a586');p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,default=None);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=20260918);a=p.parse_args()
 suite=a.suite.resolve();rows=list(csv.DictReader((suite/'summary.csv').open()))
 selected=[r for r in rows if r['variant'] in {'abc_a','abc_c','abc_d','abc_e'} and r['status']=='complete']
 if len(selected)!=4:raise RuntimeError('Expected complete A/C/D/E rows in the suite summary')
 checkpoints=[]
 for row in selected:
  run=Path(row['run_dir']);checkpoint=run/'checkpoints/best.pt'
  if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
  checkpoints.append(checkpoint)
 test_npz=Path(json_load(suite/'plan.json')['datasets']['test']['path']).resolve()
 output=(a.output or suite/'mask_panel_20260918.json').resolve()
 if output.exists():raise FileExistsError(output)
 command=[sys.executable,str(ROOT/'scripts/v24/evaluate_mask_panel.py'),'--checkpoints',*map(str,checkpoints),'--test-npz',str(test_npz),'--device',a.device,'--batch-size',str(a.batch_size),'--seed',str(a.seed),'--output',str(output)]
 env=dict(os.environ)
 if a.device=='cuda':env.setdefault('CUDA_VISIBLE_DEVICES','0')
 print('Running one checkpoint at a time on',a.device,'; output=',output)
 return subprocess.call(command,cwd=ROOT,env=env)

def json_load(path):
 import json
 return json.loads(Path(path).read_text())
if __name__=='__main__':sys.exit(main())
