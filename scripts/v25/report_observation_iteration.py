"""Complete, evidence-bounded report; called only after all frozen evaluations."""
import json
from pathlib import Path
from datetime import datetime
from run_b3_c3 import load
from observation_iteration_config import METHODS,CATEGORIES,category_metrics

def export_report(suite,payload,jobs,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(payload['project_root']);stamp=datetime.now().strftime('%Y%m%d')
    title=f'{stamp}_v25_I42-I53观测驱动多轮优化_完整分析报告'
    path=root/'experments_report'/f'{title}.md';assets=path.parent/(path.stem+'_assets');assets.mkdir(parents=True,exist_ok=True)
    lines=[f'# {title}',f'冻结套件：`{suite}`。配置、源码、初始化和选择记录均以本套件归档为准。',
           '## 1. 具体方法与复现配置','|组|直接对照|具体改动|','|---|---|---|']
    lines += [f'|{n}|{ref}|{desc}|' for n,(ref,desc) in METHODS.items()]
    lines += [f"I53实际采用 **{selection['selected']}**；仅由验证划分开发协议决定。资格：{selection['qualified_for_replication']}。",
        '共同设置：TaxiBJ真实F/M/C、双通道12步、混合四类mask 0.4、共享八专家每轮Top-2，100 epoch、有效batch32、AdamW 1e-3余弦至3e-4、每5轮ID验证选best。前后缺失F损失0.5/0.5，可见F重建0.1，可见M/C一致性0.1，候选均衡0.01。GPU1串行。',
        f"详细机制、代数依据及相近工作边界：[冻结方法说明]({suite/'source_snapshot/scripts/v25/README_OBSERVATION_ITERATION.md'})。",
        f"复现入口：`python scripts/v25/run_observation_iteration.py --suite {suite}`。源文件摘要见plan.json，初始化核对见initialization_audit.json，模型选择证据见selection.json。",
        '## 2. 完整结果、成本与诊断','### 完成情况与每套测试',
        '|组|epoch/best|参数/可训练参数|训练小时|训练峰值GiB|协议|MAE|RMSE|','|---|---|---|---|---|---|---|---|']
    reference=load(suite/'evaluations/taxibj_I37_seed7.json');records={};audit=load(suite/'initialization_audit.json');diagnostics={}
    for name,j in jobs.items():
        r=load(suite/'results'/f'{name}.json');e=load(suite/'evaluations'/f'{name}.json')
        if r['status']!='finished' or e['status']!='finished' or r['completed_epochs']!=100:raise RuntimeError('Incomplete report input '+name)
        records[name]=e;run=Path(r['run_dir']);history=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines() if l.strip()]
        key=f"I53_from_{selection['selected']}" if j['method']=='I53' else name
        a=audit[key];peak=max(row.get('perf',{}).get('peak_memory_gb',0) for row in history)
        for protocol,item in e['sets'].items():
            v=item['metrics'];lines.append(f"|{j['method']} seed{j['config']['seed']}|{r['completed_epochs']}/{r['best_epoch']}|{a['parameters']}/{a['trainable_parameters']}|{r['total_time_sec']/3600:.3f}|{peak:.3f}|{protocol}|{v['mae']:.6f}|{v['rmse']:.6f}|")
        fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].plot([h['epoch'] for h in history],[h['train']['loss'] for h in history]);axes[0].set_title('Train loss')
        valid=[h for h in history if h.get('val')];axes[1].plot([h['epoch'] for h in valid],[h['val']['mae'] for h in valid]);axes[1].set_title('ID validation MAE')
        for ax in axes:ax.set_xlabel('epoch')
        fig.tight_layout();fig.savefig(assets/f'{name}.png',dpi=160);plt.close(fig)
        diagnostics[name]={p:x['metrics'] for p,x in e['sets'].items()}
    for protocol,item in reference['sets'].items():
        lines.append(f"|I37 seed7|100/{reference['best_epoch']}|见参考归档|—|—|{protocol}|{item['metrics']['mae']:.6f}|{item['metrics']['rmse']:.6f}|")
    lines += ['### 曲线与训练过程']
    for name,j in jobs.items():
        r=load(suite/'results'/f'{name}.json');run=Path(r['run_dir']);h=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines() if l.strip()]
        lines += [f"{j['method']}：训练loss {h[0]['train']['loss']:.5f}→{h[-1]['train']['loss']:.5f}；best ID验证MAE {r['best_val_mae']:.6f}，epoch {r['best_epoch']}。",
                  f'![{name}]({assets.name}/{name}.png)',f'配置和日志：[{name}]({run})。']
    # Full metric objects include exact raw/post and constrained/unconstrained counts,
    # routing paths, observation residuals, pool and communication diagnostics.
    diagnostic_path=assets/'complete_diagnostics.json';diagnostic_path.write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2))
    lines += [f'完整六协议逐轮与区域诊断（含有效计数）：[JSON]({assets.name}/complete_diagnostics.json)。空区域MAE应读作“—”，不得把空计数的数值当作性能。',
              '### 开发协议选择记录', '```json', json.dumps(selection,ensure_ascii=False,indent=2),'```',
              '### 新mask确认协议（独立列示）','|模型|协议|MAE|RMSE|','|---|---|---|---|']
    for file in sorted((suite/'confirmation').glob('*.json')):
        ev=load(file)
        for name,s in ev['sets'].items():lines.append(f"|{file.stem}|{name}|{s['metrics']['mae']:.6f}|{s['metrics']['rmse']:.6f}|")
    lines += ['## 3. 分析、证据边界与建议','### 预定推荐门槛',
              '|组|同seed参考|ID代价%|OOD三类等权MAE变化%|最差类别变化%|同时达标|','|---|---|---|---|---|---|']
    for name,j in jobs.items():
        r=records[name];ref=records['taxibj_I52_seed17'] if j['config']['seed']==17 else reference
        idratio=r['sets']['in_distribution']['metrics']['mae']/ref['sets']['in_distribution']['metrics']['mae']
        ratios=[a/b for a,b in zip(category_metrics(r,'mae'),category_metrics(ref,'mae'))];avg=sum(ratios)/3;worst=max(ratios)
        ok=idratio<=1.03 and avg<=.97 and worst<=1.05
        lines.append(f"|{j['method']}|{'I52' if j['config']['seed']==17 else 'I37'}|{(idratio-1)*100:+.3f}|{(avg-1)*100:+.3f}|{(worst-1)*100:+.3f}|{'是' if ok else '否'}|")
    lines += ['### 机制对照与负结果','以下按预先指定直接对照计算，变化为(新−参考)/参考，负值为改善。',
              '|组|直接对照|ID MAE变化%|OOD等权类别MAE比变化%|','|---|---|---|---|']
    for name,j in jobs.items():
        method=j['method'];parent=METHODS[method][0]
        if method in ('I52','I53'):continue
        ref=reference if parent=='I37' else records[f'taxibj_{parent}_seed7'];r=records[name]
        delta=r['sets']['in_distribution']['metrics']['mae']/ref['sets']['in_distribution']['metrics']['mae']-1
        ood=sum(a/b for a,b in zip(category_metrics(r,'mae'),category_metrics(ref,'mae')))/3-1
        lines.append(f'|{method}|{parent}|{delta*100:+.3f}|{ood*100:+.3f}|')
    lines += ['门槛未达标的组均完整列示，不能因局部最好而宣称全面优化。若无同时达标者，本批结论是未找到满足预定要求的改进，应保留I37候选。',
        'I53先在验证开发协议选定，再开启本批测试。表中测试排名不用于改换I53。两种训练seed分别展示，mask复测不是训练seed复验，新mask不是新数据集；不宣称统计显著性或全局最优。',
        '代数验收只说明消息分解和隐藏区域状态保持性质；非线性编码、专家及解码后不自动保证观测约束或误差单调下降。最终硬约束来自原学习投影。平均指标改善不能单独证明去重复或专家职责解释。',
        '后续优先检查观测残差、历史创新、通信幅度和无约束区域的实际变化是否与对照假设一致；再对有证据的方向扩展独立训练seed和数据集。不根据本批测试临时增加组合。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8');(suite/'report_path.txt').write_text(str(path));return path
