#!/usr/bin/env python3
"""Wait for the untouched replay; validate/perf-test/fully replay the candidate."""
import argparse,fcntl,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
FEATURES=['skip_redundant_init','reuse_fine_support','no_grad_intermediate','compact_outputs','cache_constants','pack_scalar_logs','skip_last_history']
load=lambda p:json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(suite,status,**kw):write(suite/'queue_state.json',dict(status=status,time=time.time(),**kw))
def run(cmd,log,env,cwd):
 log.parent.mkdir(exist_ok=True,parents=True)
 with log.open('ab') as f:
  child=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  try:
   while chunk:=os.read(child.stdout.fileno(),4096):f.write(chunk);f.flush();sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
   code=child.wait()
  except BaseException:child.terminate();child.wait();raise
 if code:raise RuntimeError(f'Child exited {code}; see {log}')
def equal_trace(reference,candidate):
 diffs=[]
 for case,expected in reference['cases'].items():
  actual=candidate.get('cases',{}).get(case)
  if actual!=expected:
   fields=sorted({k for x,y in zip(expected,actual or []) for k in x if x[k]!=y.get(k)})
   diffs.append(dict(case=case,fields=fields or ['trace_length_or_case']))
 return diffs

def preliminary_report(suite,p,selection,benchmark):
 from datetime import datetime
 from zoneinfo import ZoneInfo
 date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d');root=Path(p['project_root']);path=root/'experments_report'/f'{date}_best-perf_I37性能与严格等价验证_阶段报告.md'
 text=f'''# {date}_best-perf_I37性能与严格等价验证_阶段报告

## 1. 方法与配置

原版为best提交{p['base_commit']}，候选为best-perf提交{p['candidate_commit']}。方法/损失/初始化/数据不变，训练诊断首/末及每20批采集，评估完整。候选可选七项执行优化，源码冻结，GPU1等原版复跑完成后独占。

## 2. 检查与实测

CPU11项通过；GPU逐项、组合、AMP完整恢复轨迹与原实现逐位核对。选择记录如下；失败项没有静默合入。

```json
{json.dumps(selection,ensure_ascii=False,indent=2)}
```

固定实际数据batch32，原best权重仅用于微基准，每组3次、每次5批预热＋20批测量；正式完整复跑从未训练初始化开始。

```json
{json.dumps(benchmark,ensure_ascii=False,indent=2)}
```

## 3. 边界

此为阶段报告。未完成优化版100轮前，不宣称最终逐位复现；微基准不能代替整轮和整次训练成本。检查点格式和原best保持，数据/权重不覆盖。
'''
 path.write_text(text);write(suite/'stage_report.json',dict(path=str(path)));return path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,required=True);args=parser.parse_args();suite=args.suite.resolve();p=load(suite/'plan.json');candidate=Path(p['candidate_source']);reference=Path(p['reference_source']);locks=[]
 try:
  for relative,stamp in p['candidate_sha256'].items():
   if sha(candidate/relative)!=stamp:raise RuntimeError('Candidate frozen source changed '+relative)
  if load(suite/'acceptance/cpu.json')['status']!='passed':raise RuntimeError('CPU acceptance required')
  while True:
   parent=load(Path(p['reference_replay'])/'queue_state.json')
   if parent['status']=='finished':
    if parent.get('verdict')!='exact_reproduction':raise RuntimeError('Unoptimized replay was not exact; inspect it before optimizing')
    break
   if parent['status']=='failed':raise RuntimeError('Unoptimized replay failed; do not start GPU work')
   state(suite,'waiting_for_reference',reference_status=parent['status']);time.sleep(15)
  for lockpath in [suite/'queue.lock',Path(p['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock']:
   lockpath.parent.mkdir(exist_ok=True,parents=True);f=lockpath.open('a')
   while True:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f);break
    except BlockingIOError:state(suite,'waiting_for_gpu1_lock');time.sleep(15)
  # Refuse a contaminated performance measurement; never terminate other jobs.
  uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();gpu1=uuids[1].strip()
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if any(line.split(',')[0].strip()==gpu1 for line in processes.splitlines()):raise RuntimeError('GPU1 has an unrelated compute process; benchmark not started')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/best_perf_mpl')
  helper=candidate/'scripts/performance_probe.py';refout=suite/'acceptance/gpu_reference/trace.json'
  def probe(label,features,source,compact=True):
   target=suite/'acceptance'/label/'trace.json'
   if not target.exists():
    cmd=[sys.executable,'-u',str(helper),'--source',str(source),'--data-dir',p['data_dir'],'--output',str(target),'--features',features]
    if compact:cmd+=['--compact']
    state(suite,'gpu_strict_checks',active=label);run(cmd,suite/'logs'/f'{label}.log',env,suite)
   return load(target)
  original=probe('gpu_reference','none',reference,False);accepted=[];rejected={}
  diagnostic=probe('gpu_diagnostics','diagnostics',candidate)
  if equal_trace(original,diagnostic):raise RuntimeError('Diagnostic-only path failed strict equivalence')
  for key in FEATURES:
   item=probe('gpu_'+key,key,candidate);diffs=equal_trace(original,item)
   if diffs:rejected[key]=diffs
   else:accepted.append(key)
  def spec(keys):return ','.join(keys) if keys else 'diagnostics'
  combined=probe('gpu_combined',spec(accepted),candidate)
  if equal_trace(original,combined):
   retained=[]
   for key in accepted:
    candidate_keys=retained+[key];diffs=equal_trace(original,probe('gpu_incremental_'+key,spec(candidate_keys),candidate))
    if diffs:rejected[key]=diffs
    else:retained=candidate_keys
   accepted=retained
   if equal_trace(original,probe('gpu_final',spec(accepted),candidate)):raise RuntimeError('Final combination is not strictly equivalent')
  selection=dict(accepted=accepted,rejected=rejected,diagnostic_interval=20,strict=True)
  write(suite/'selection.json',selection)
  # Freeze final configuration before any timing results.
  cfg=load(candidate/'configs/i37_taxibj.json');cfg['train']['performance']=dict(enabled=True,diagnostic_interval=20,**{k:k in accepted for k in FEATURES});write(suite/'selected_config.json',cfg)
  weights=Path(p['reference_replay'])/'run/checkpoints/best.pth';bench={}
  for label,source,features in [('baseline',reference,'none'),('optimized',candidate,spec(accepted))]:
   target=suite/'benchmark'/label/'result.json'
   if not target.exists():
    state(suite,'benchmark',active=label);run([sys.executable,'-u',str(candidate/'scripts/performance_benchmark.py'),'--source',str(source),'--data-dir',p['data_dir'],'--output',str(target),'--features',features,'--weights',str(weights)],suite/'logs'/f'benchmark_{label}.log',env,suite)
   bench[label]=load(target)
  for x,y in zip(bench['baseline']['trials'],bench['optimized']['trials']):
   for key in ['model_sha256','optimizer_sha256','rng_sha256','primary']:
    if x[key]!=y[key]:raise RuntimeError('Engine benchmark is not bitwise equivalent: '+key)
  median=lambda label,key:statistics.median(row[key] for row in bench[label]['trials'])
  speed=median('optimized','seconds_per_step')/median('baseline','seconds_per_step');memory=median('optimized','peak_allocated_gib')/median('baseline','peak_allocated_gib')
  benefit=(speed<=.97 and memory<=1.02) or (memory<=.95 and speed<=1.02)
  summary=dict(step_time_ratio=speed,allocated_ratio=memory,meets_predeclared_benefit=benefit,baseline={k:median('baseline',k) for k in ['seconds_per_step','peak_allocated_gib','peak_reserved_gib']},optimized={k:median('optimized',k) for k in ['seconds_per_step','peak_allocated_gib','peak_reserved_gib']})
  write(suite/'benchmark_summary.json',summary);stage=preliminary_report(suite,p,selection,summary)
  if not benefit:
   state(suite,'finished_no_benefit',report=str(stage));return
  if not (suite/'run/result.json').exists():
   state(suite,'training_and_evaluation')
   run([sys.executable,'-u',str(candidate/'scripts/train.py'),'--data-dir',p['data_dir'],'--run-dir',str(suite/'run'),'--config',str(suite/'selected_config.json'),'--gpu','1','--threads','2','--evaluate'],suite/'logs/full_replay.log',env,candidate)
  state(suite,'comparison_and_report')
  from performance_report import export_report
  path,verdict=export_report(suite,p,selection,summary)
  state(suite,'finished',verdict=verdict,report=str(path))
 except BaseException as error:
  state(suite,'failed',error=repr(error));raise
if __name__=='__main__':main()
