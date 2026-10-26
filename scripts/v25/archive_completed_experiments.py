#!/usr/bin/env python3
"""Export lightweight, explicitly reduced evidence; never train or modify outputs."""
import csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from observation_iteration_config import category_metrics
SPECS={
 'i42_i53':('observation_iteration/1d6eb79a66c77e49','20261022_v25_I42-I53观测驱动多轮优化_完整分析报告.md'),
 'i54_i63':('input_refinement/ad274d0529acb794','20261022_v25_I54-I63输入整理与三种子复验_完整分析报告.md'),
 'i64_i75':('final_input/a14c08aae45f4052','20261025_v25_I64-I75最后一轮seed7输入优化_复核分析报告.md')}
load=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def keep(k):
 return k in ['loss','mae','rmse','wape','metric_missing_count','forward_ms_per_sample_per_rank','coe_expert_grid_equivalents'] or k.startswith(('combination_','coe_scale_execution_path_','coe_v25_consistency_','coe_v25_observed_','coe_input_step','coe_x_step'))
def export():
 destination=ROOT/'experiment_records/v25';index={}
 for batch,(relative,report_name) in SPECS.items():
  suite=ROOT/'outputs/v25/experiments'/relative;target=destination/batch;target.mkdir(parents=True,exist_ok=True)
  plan=load(suite/'plan.json');original_hashes={}
  for name in ['plan.json','resolved_jobs.json','resolved_templates.json','selection.json','initialization_audit.json','queue_state.json']:
   source=suite/name
   if source.exists():write(target/name,load(source));original_hashes[name]=sha(source)
  for job in (suite/'jobs').glob('*.json'):write(target/'jobs'/job.name,load(job));original_hashes['jobs/'+job.name]=sha(job)
  for receipt in (suite/'results').glob('*.json'):
   r=load(receipt);write(target/'results'/receipt.name,r);original_hashes['results/'+receipt.name]=sha(receipt)
   run=Path(r['run_dir']);history=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines() if s.strip()]
   compact=[]
   for h in history:
    compact.append(dict(epoch=h['epoch'],is_best=h['is_best'],perf=h['perf'],
     train={k:v for k,v in h['train'].items() if k in ['loss','mae','rmse','lr']},
     val={k:v for k,v in h['val'].items() if k in ['loss','mae','rmse']} if h.get('val') else None))
   write(target/'curves'/receipt.name,compact)
  groups={}
  for group in ['development','evaluations','confirmation']:
   values={}
   for file in (suite/group).glob('*.json'):
    raw=load(file);copy={k:v for k,v in raw.items() if k!='sets'}
    copy['archive']=dict(reduced_metrics=True,source_sha256=sha(file),source_bytes=file.stat().st_size,
     note='Primary metrics, area/counts, observed consistency, scale paths, input/history diagnostics retained. Dense conditional expert tables omitted.')
    copy['sets']={name:{**{k:v for k,v in row.items() if k!='metrics'},'metrics':{k:v for k,v in row['metrics'].items() if keep(k)}} for name,row in raw['sets'].items()}
    write(target/group/file.name,copy);values[file.stem]=copy
   groups[group]=values
  formal=ROOT/'experments_report'/report_name
  reports=[formal]
  if batch=='i64_i75':reports.append(ROOT/'experments_report/20261024_v25_I64-I75最后一轮seed7输入优化_完整分析报告.md')
  for report in reports:
   if not report.exists():continue
   for match in re.finditer(r'\]\(([^)]+)\)',report.read_text()):
    link=match.group(1).strip();linked=Path(link) if link.startswith('/') else report.parent/link
    if not linked.is_file():continue
    if linked.suffix in ('.png','.pdf'):
     out=target/'figures'/linked.name
     out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(linked.read_bytes())
  readme=['# '+batch+'：已完成实验档案','',
   '这里保留冻结计划、实际配置、完成回执、曲线和三组评价。所有文件由已有产物只读导出，不会重新训练。',
   '数据、best/last权重、完整训练/GPU日志和冻结源码大副本仍在原服务器，不在Git中。',
   '评价JSON有明确reduced_metrics标记，保留主要误差、有效计数、投影、输入与尺度路径，省略密集条件专家表；source_sha256可与原文件核对。',
   'plan/config/receipt中的绝对路径是原执行记录，不能当作新机器上的可用路径。独立I37复现请使用仓库best分支。','',
   f'原套件：`outputs/v25/experiments/{relative}`。',f'完整分析报告：[报告](../../../experments_report/{report_name})。','']
  (target/'README.md').write_text('\n'.join(readme))
  write(target/'archive_provenance.json',dict(original_suite=str(suite.relative_to(ROOT)),original_files_sha256=original_hashes,
   evaluation_groups={g:len(v) for g,v in groups.items()},metric_archive_is_reduced=True,checkpoints_included=False))
  index[batch]=dict(path=batch,original_suite=relative,new_trainings=len(list((suite/'results').glob('*.json'))),
   evaluated_models={g:len(v) for g,v in groups.items()},report=str(formal.relative_to(ROOT)))
  with (target/'summary.csv').open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['variant','id_mae','id_rmse','binary_mae','geometry_mae','triple_mae','ood_three_category_mean'])
   for name,r in sorted(groups['evaluations'].items()):
    m=r['sets']['in_distribution']['metrics'];c=category_metrics(r,'mae');writer.writerow([name,m['mae'],m['rmse'],*c,sum(c)/3])
 write(destination/'index.json',dict(baseline='I37',decision_date='2026-10-26',batches=index))
 baseline_suite=ROOT/'outputs/v25/experiments/taxibj_input_communication/f84c43ccee2d4261'
 original_job=load(baseline_suite/'jobs/taxibj_I37_seed7.json')
 write(destination/'i37/job.original.json',original_job)
 receipt=load(baseline_suite/'results/taxibj_I37_seed7.json');write(destination/'i37/result.original.json',receipt)
 cfg=original_job['config'];write(ROOT/'configs/v25/i37_baseline.json',cfg)
 baseline_eval=load(ROOT/'outputs/v25/experiments/final_input/a14c08aae45f4052/evaluations/taxibj_I37_seed7.json')
 write(destination/'i37/reference_metrics.json',{name:{k:row['metrics'][k] for k in ['mae','rmse','metric_missing_count']} for name,row in baseline_eval['sets'].items()})
 print(json.dumps(index,ensure_ascii=False,indent=2))
if __name__=='__main__':export()
