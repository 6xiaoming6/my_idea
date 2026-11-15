#!/usr/bin/env python3
"""Freeze an immutable source snapshot and all ten jobs before launch."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.deterministic import state_hash
from release_utils import load,write,sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
 suite=a.suite.resolve();project=a.project.resolve();source=suite/'source'
 if (suite/'plan.json').exists() or source.exists():raise FileExistsError('Suite is already frozen; resume it instead')
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit implementation before freezing')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 data=project/'data/TaxiBJ/v25_observed_multiscale/73fe84df49ca05de'
 if load(suite/'preflight_gpu/results.json')['status']!='passed':raise RuntimeError('GPU preflight not passed')
 if load(suite/'equivalence_gpu/result.json')['status']!='passed':raise RuntimeError('Reuse equivalence check missing')
 for name in ('GS06','GS10'):
  if load(suite/f'worst_{name}/result.json')['status']!='passed':raise RuntimeError('Worst-path preflight missing: '+name)
 if not (suite/'cpu_tests.log').read_text().rstrip().endswith('OK'):raise RuntimeError('CPU tests not passed')
 names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines();source.mkdir()
 for name in names:
  src=ROOT/name;dst=source/name
  if not src.is_file():continue
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 reference=suite/'reference_I37';reference.mkdir()
 old=project/'outputs/v25/experiments/i37_performance/d920258a_seed7_20261026/run'
 for name in ['config.json','result.json','evaluation.json','training_metadata.json','initialization.json','runtime.json','run_provenance.json','logs/metrics.jsonl']:
  dst=reference/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,dst)
 old_data=load(reference/'run_provenance.json')['data_sha256']
 for split in ('train','val','test'):
  if sha256(data/f'taxibj_{split}.npz')!=old_data[split]:raise RuntimeError('Reference data changed')
 jobs=load(source/'configs/latent_size/suite.json');initializations={};initial=suite/'initializations';initial.mkdir()
 torch.set_num_threads(2)
 torch.manual_seed(7);common=DualBranchSTImputer.from_config(load(source/'configs/i37_performance.json')).state_dict()
 for job in jobs:
  torch.manual_seed(7);model=DualBranchSTImputer.from_config(load(source/job['config']));state=model.state_dict()
  for name,value in common.items():
   if not torch.equal(value,state[name]):raise RuntimeError('Common initialization mismatch: '+name)
  record=next(r for r in load(suite/'preflight_gpu/results.json')['rows'] if r['name']==job['name'])
  fingerprint=state_hash(state)
  if fingerprint!=record['initial_hash']:raise RuntimeError('GPU preflight initial state mismatch')
  path=initial/f'{job["name"]}.pth';torch.save(dict(model=state,seed=7,sha256=fingerprint),path)
  initializations[job['name']]=dict(path=str(path),state_sha256=fingerprint,params=record['params'],trainable_params=record['trainable_params'])
 shutil.copy2(project/'outputs/size_preflight_gpu.log',suite/'preflight_gpu/console.log')
 # Reuse only the verified 16x16 run. Never resume the old multi-variant queue.
 old_suite=project/'outputs/v25/experiments/latent_space/20261114_seed7'
 reused=jobs[2]
 reused.update(run_dir=str(old_suite/'runs/L01'),source_override=str(old_suite/'source'),config_override=str(old_suite/'source/configs/latent_space/L01.json'),variant_name='L01',resume_note='Verified exact equivalent; resume original complete epoch80 checkpoint')
 jobs=[reused]+jobs[:2]+jobs[3:]
 plan=dict(version='v25_latent_size',base_commit='3ab02340',source_commit=commit,source=str(source),reference=str(reference),python=sys.executable,data_dir=str(data),report_dir=str(project/'experments_report'),physical_gpu=1,jobs=jobs,initializations=initializations,ranking='ID <= I37 * 1.03; OOD category MAE ratio, OOD RMSE ratio, ID MAE; disclose all')
 write(suite/'plan.json',plan)
 files=[f for f in source.rglob('*') if f.is_file()]+[f for f in reference.rglob('*') if f.is_file()]+list(initial.glob('*.pth'))+[data/f'taxibj_{s}.npz' for s in ('train','val','test')]+[suite/'plan.json',suite/'preflight_gpu/results.json',suite/'cpu_tests.log']
 files += [f for f in (old_suite/'source').rglob('*') if f.is_file() and '__pycache__' not in str(f)]
 files += [suite/'equivalence_gpu/result.json']+[suite/f'worst_{name}/result.json' for name in ('GS06','GS10')]
 write(suite/'manifest.json',dict(files={str(f):sha256(f) for f in files}))
 command=f'#!/usr/bin/env bash\nset -uo pipefail\nexport OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n{sys.executable} -u {source}/scripts/queue_sizes.py --suite {suite} 2>&1 | tee -a {suite}/queue.log\ncode=${{PIPESTATUS[0]}}\nprintf "%s\\n" "$code" > {suite}/exit_code\nexit "$code"\n'
 (suite/'start_queue.sh').write_text(command)
 print(json.dumps(dict(source=source.as_posix(),commit=commit,jobs=len(jobs)),indent=2))
if __name__=='__main__':main()
