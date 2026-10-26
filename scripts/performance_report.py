"""Use the frozen replay reporter, adding optimizer/RNG and performance evidence."""
import importlib.util,json,statistics
from pathlib import Path
load=lambda p:json.loads(Path(p).read_text())
def export_report(suite,plan,selection,benchmark):
 import torch
 parent=Path(plan['reference_replay']);project=Path(plan['project_root'])
 # Adapt the existing reporter's legacy receipt layout without modifying any run.
 alias=suite/'reference/results';alias.mkdir(exist_ok=True,parents=True)
 (alias/'taxibj_I37_seed7.json').write_text((parent/'run/result.json').read_text())
 baseline=load(parent/'plan.json')
 bridge=dict(project_root=str(suite/'report_build'),release_commit=plan['candidate_commit'],source=plan['candidate_source'],reference_run=str(parent/'run'),reference_suite=str(suite/'reference'),reference_evaluation=str(parent/'run/evaluation.json'),data_dir=plan['data_dir'],initialization_expected=baseline['initialization_expected'],comparison=dict(metrics_absolute_tolerance=0.,rmse_absolute_tolerance=0.))
 # The helper only reads plan.json; use a separate adapter directory with links
 # to this run, keeping the original performance plan immutable.
 adapter=suite/'report_adapter';adapter.mkdir(exist_ok=True)
 (adapter/'plan.json').write_text(json.dumps(bridge,indent=2))
 if not (adapter/'run').exists():(adapter/'run').symlink_to(suite/'run',target_is_directory=True)
 helper=Path(plan['comparison_reporter']);spec=importlib.util.spec_from_file_location('frozen_i37_reporter',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 report_path,verdict=module.report(adapter)
 comparison=load(adapter/'comparison.json');extra={}
 def equal(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  try:
   import numpy as np
   if isinstance(a,np.ndarray):return np.array_equal(a,b)
  except ImportError:pass
  return a==b
 for kind in ['best','last']:
  a=torch.load(parent/'run/checkpoints'/f'{kind}.pth',map_location='cpu',weights_only=False);b=torch.load(suite/'run/checkpoints'/f'{kind}.pth',map_location='cpu',weights_only=False)
  extra[kind]={key:equal(a[key],b[key]) for key in ['optimizer','scheduler','scaler','rng_states']}
 if not all(v for row in extra.values() for v in row.values()):verdict='optimizer_or_rng_mismatch'
 comparison['complete_training_state_equal']=extra;comparison['performance']=benchmark;comparison['selection']=selection;comparison['verdict']=verdict
 (suite/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
 text=report_path.read_text();text=text.replace('best分支I37同条件复跑','best-perf严格等价与性能复跑').replace('原I37','未优化best复跑')
 text=text.replace('## 3. 分析、边界及结论',f'''### 性能与附加状态验收

训练诊断按首/末/每20批采集；验证及测试完整。源码/参数形状/训练任务不变，逐项检查后拒绝的选项没有纳入最终配置。

```json
{json.dumps(dict(selection=selection,benchmark=benchmark,complete_state=extra),ensure_ascii=False,indent=2)}
```

微基准固定batch32、预热5批、测量20批、3次重复，不能替代本次100轮训练实测成本。候选完整训练耗时与未优化复跑耗时见上表；预检权重与正式训练初始化分离。源best分支未被改写。

## 3. 分析、边界及结论''')
 old=module.load(adapter/'comparison.json')['verdict'];text=text.replace('**'+old+'**','**'+verdict+'**')
 if verdict!='exact_reproduction':text+='\n严格验收未全部通过，不能合入原best。\n'
 # Keep the helper's dated asset links; create a distinct performance report.
 path=project/'experments_report'/report_path.name.replace('best分支I37同条件复跑','best-perf_I37严格等价与性能')
 import shutil
 assets=path.parent/(path.stem+'_assets');assets.mkdir(exist_ok=True,parents=True)
 original_assets=report_path.parent/(report_path.stem+'_assets')
 for file in original_assets.iterdir():shutil.copy2(file,assets/file.name)
 text=text.replace(original_assets.name,assets.name)
 path.write_text(text);(suite/'report_path.txt').write_text(str(path)+'\n');return path,verdict
