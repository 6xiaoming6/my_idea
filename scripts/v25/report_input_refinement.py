"""Formal I54-I63 report: methods, full results/diagnostics, then interpretation."""
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from run_b3_c3 import load
from input_refinement_config import METHODS, CANDIDATES, category_metrics


def score(record, reference):
    ratios=[a/b for a,b in zip(category_metrics(record,'mae'),category_metrics(reference,'mae'))]
    id_ratio=record['sets']['in_distribution']['metrics']['mae']/reference['sets']['in_distribution']['metrics']['mae']
    return id_ratio,sum(ratios)/3,max(ratios)


def export_report(suite,payload,jobs,selection):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib_i54_reports')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(payload['project_root']);date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    path=root/'experments_report'/f'{date}_v25_I54-I63输入整理与三种子复验_完整分析报告.md'
    assets=path.parent/(path.stem+'_assets');assets.mkdir(parents=True,exist_ok=True)
    references={7:'taxibj_I37_seed7',17:'taxibj_I52_seed17',27:'taxibj_I61_seed27'}
    records={name:load(suite/'evaluations'/f'{name}.json') for name in [*references.values(),*jobs]}
    for record in records.values():
        if record['status']!='finished' or len(record['sets'])!=6:raise RuntimeError('Incomplete formal evaluation')
    lines=[f'# {date}_v25_I54-I63输入整理与三种子复验_完整分析报告','',f'冻结套件：`{suite}`。','',
           '## 1. 每组具体方法与复现配置','','|组|直接对照|实际改动|','|---|---|---|']
    lines += [f'|{n}|{parent}|{description}|' for n,(parent,description) in METHODS.items()]
    lines += ['',f"I62/I63均采用开发协议选定的 **{selection['selected']}**，分别为seed17/27；选择仅使用验证窗口。ID入选资格不等于达到最终推荐标准。",'',
        '共同设置：TaxiBJ已有2452/342/707窗口，真实F/M/C独立mask，双通道12步，F/M/C空间32/16/8。训练四种单缺失族混合0.4；原共享八专家四轮原生Top2，前三轮自由选尺度、末轮F；H0三尺度编码融合、受控差分、历史池、最终学习分配投影保持原I37。', '',
        '100 epoch、batch32、GPU1单卡串行、AMP、AdamW lr1e-3余弦至3e-4、weight_decay1e-4、grad_clip_norm1，每5轮ID验证MAE选best。前后缺失F损失0.5/0.5，可见F重建0.1、可见M/C一致性0.1、候选均衡0.01。公共未训练初始化；没有加载原I37已训练权重。所有输入变体增加参数为0，保留原参数键；某些输入通道置零后对应权重仍在参数表中，不能把总参数相同当作有效容量完全相同。','',
        'I54只替换专家输入的缺失补位。I55只替换路由两个显式值统计块：可见F均值按可见计数归一，缺失值统计置零；隐藏状态和原mask/support统计仍保留。I56组合两者。I57只在逐轮专家输入中将support/position置零；H0和路由仍有相应信息，原生观测mask继续输入。','',
        'I58使用独立于模型预测的静态观测预算：M扣可见F，C扣可见M总量及其他可见F，剩余预算除剩余F个数；无剩余区域的原生尺度槽置零，mask/support一致更新。H0和最终投影仍使用原始观测。I59只向专家缺失槽加入detach的当前预测，以0.1倍可见F/M/C密度RMS乘tanh限幅；预测不改变mask。I60预先组合I57/I58，不含I59反馈。','',
        '训练种子7/17/27；新开发mask基种子2026102201–06，验证划分；新确认mask基种子2026102211–16，测试划分。原六套测试协议保持既有mask基种子。三尺度mask独立，保留原split偏移；新mask不是新训练种子/数据集。', '',
        '每组完成后立即运行开发ID/OOD；先固定入选方案，再训练I62/I63，之后开放全部正式测试。开发ID≤原I37同协议103%，按三类等权相对OOD MAE排序，同分按OOD RMSE/ID/参数/编号；无合格者按最小ID超限选择并标记。', '',
        f"完整冻结定义：[README]({suite/'source_snapshot/scripts/v25/README_INPUT_REFINEMENT.md'})；源码、数据摘要、参考文件摘要见plan.json，初始化见initialization_audit.json。",'',
        f'恢复命令：`python scripts/v25/run_input_refinement.py --suite {suite}`。','',
        '|模型|冻结作业配置|原始run|','|---|---|---|']
    for name,j in jobs.items():
        r=load(suite/'results'/f'{name}.json')
        lines.append(f"|{j['method']} seed{j['config']['seed']}|[{name}]({suite/'jobs'/f'{name}.json'})|[日志与best/last]({r['run_dir']})|")
    lines += ['','## 2. 完整结果、曲线与诊断','',
              '|模型|ID MAE|ID RMSE|二元MAE|形态MAE|三元MAE|OOD均值|同seed ID变化%|同seed OOD变化%|最差类变化%|达标|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    summaries=[]
    for name,record in records.items():
        seed=int(name.rsplit('seed',1)[1]);ref=records[references[seed]];idratio,ood,worst=score(record,ref)
        category=category_metrics(record,'mae');metrics=record['sets']['in_distribution']['metrics']
        eligible=idratio<=1.03 and ood<=.97 and worst<=1.05;is_reference=name==references[seed]
        lines.append(f"|{name}|{metrics['mae']:.6f}|{metrics['rmse']:.6f}|{category[0]:.6f}|{category[1]:.6f}|{category[2]:.6f}|{sum(category)/3:.6f}|{100*(idratio-1):+.3f}|{100*(ood-1):+.3f}|{100*(worst-1):+.3f}|{'参考' if is_reference else '是' if eligible else '否'}|")
        summaries.append(dict(variant=name,seed=seed,id_mae=metrics['mae'],id_rmse=metrics['rmse'],category_mae=category,
                              id_ratio=idratio,ood_ratio=ood,worst_ratio=worst,meets_rule=eligible,is_reference=is_reference))
    lines += ['','二元/形态各先合并两次mask，OOD均值是三类MAE平均；相对OOD变化是类别比值平均。两者不是同一百分比口径。seed7/17/27分别对照原I37/I52/I61。','',
              '### 原六套完整测试','','|模型|协议|MAE|RMSE|','|---|---|---:|---:|']
    for name,record in records.items():
        for protocol,row in record['sets'].items():
            lines.append(f"|{name}|{protocol}|{row['metrics']['mae']:.6f}|{row['metrics']['rmse']:.6f}|")
    lines += ['','### 训练成本、提交记录和曲线','','|模型|完成/best|最佳ID验证MAE|参数/可训练参数|训练小时|峰值GiB|best/last|GPU日志|','|---|---|---:|---|---:|---:|---|---|']
    plots=[]
    for name,j in jobs.items():
        r=load(suite/'results'/f'{name}.json');run=Path(r['run_dir']);meta=load(run/'training_metadata.json')
        h=[__import__('json').loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines() if s.strip()]
        if len(h)!=100 or r['completed_epochs']!=100:raise RuntimeError('Incomplete training record '+name)
        present=all((run/'checkpoints'/n).exists() for n in ['best.pth','last.pth'])
        if not present:raise RuntimeError('Missing checkpoint '+name)
        lines.append(f"|{name}|100/{r['best_epoch']}|{r['best_val_mae']:.6f}|{meta['total_params']}/{meta['trainable_params']}|{r['total_time_sec']/3600:.3f}|{max(x['perf']['peak_memory_gb'] for x in h):.3f}|有|{'有' if (run/'logs/gpu_status.jsonl').exists() else '缺失'}|")
        fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].plot([x['epoch'] for x in h],[x['train']['loss'] for x in h]);v=[x for x in h if x.get('val')]
        axes[1].plot([x['epoch'] for x in v],[x['val']['mae'] for x in v]);axes[0].set_title('Train loss');axes[1].set_title('ID validation MAE')
        for ax in axes:ax.set_xlabel('epoch');ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(assets/f'{name}.png',dpi=160);plt.close(fig)
        plots += [f"{name}：训练loss {h[0]['train']['loss']:.4f}→{h[-1]['train']['loss']:.4f}；最后ID验证MAE {v[-1]['val']['mae']:.4f}，best在{r['best_epoch']}轮。",'',f'![{name}]({assets.name}/{name}.png)','']
    lines += ['','### 曲线','',*plots,'### 输入、路由与投影诊断','',
              '|模型|协议|原始F MAE|最终F MAE|约束区域MAE/计数|无约束区域MAE/计数|最后轮显式预测补位幅度|','|---|---|---:|---:|---|---|---:|']
    for name,record in records.items():
        for protocol in ['in_distribution','unseen_geometry','unseen_triple']:
            m=record['sets'][protocol]['metrics']
            def area(which):
                count=m.get('combination_post_'+which+'_count',0)
                return f"{m['combination_post_'+which+'_mae']:.5f}/{int(count)}" if count else '—/0'
            estimate=m.get('coe_input_step4_estimate_abs')
            lines.append(f"|{name}|{protocol}|{m['combination_raw_all_mae']:.5f}|{m['mae']:.5f}|{area('constrained')}|{area('unconstrained')}|{estimate if estimate is not None else '—'}|")
    diagnostic=assets/'complete_diagnostics.json';diagnostic.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    lines += ['',f'完整有效计数、每轮输入、历史/差分、尺度路径、专家使用和区域诊断：[JSON]({assets.name}/complete_diagnostics.json)。空区域显示“—”。原始误差使用x_hat_raw_consistency对应统计，不把投影覆盖后的最后轮预测当成原始预测。','',
              '### 开发选择记录','','```json',json.dumps(selection,ensure_ascii=False,indent=2),'```','',
              '### 新mask确认（相同测试窗口）','','|模型|协议|MAE|RMSE|','|---|---|---:|---:|']
    for file in sorted((suite/'confirmation').glob('*.json')):
        e=load(file)
        if e['status']!='finished':raise RuntimeError('Incomplete confirmation')
        for protocol,row in e['sets'].items():lines.append(f"|{file.stem}|{protocol}|{row['metrics']['mae']:.6f}|{row['metrics']['rmse']:.6f}|")
    lines += ['','## 3. 分析、边界与建议','', '### 逐组直接对照','',
              '变化为新/参考−1；负值改善。I56/I60含组合，各自列出组成对照，不能只与较弱的一组比较。','',
              '|方法|直接对照|ID变化%|OOD类别比变化%|','|---|---|---:|---:|']
    parents={'I54':['I37'],'I55':['I37'],'I56':['I54','I55'],'I57':['I56'],'I58':['I56'],'I59':['I56'],'I60':['I57','I58']}
    for n in CANDIDATES:
        record=records[f'taxibj_{n}_seed7']
        for parent in parents[n]:
            ratios=score(record,records[f'taxibj_{parent}_seed7'])
            lines.append(f'|{n}|{parent}|{100*(ratios[0]-1):+.3f}|{100*(ratios[1]-1):+.3f}|')
    candidates=[s for s in summaries if s['seed']==7 and not s['is_reference']]
    qualified=[s for s in candidates if s['meets_rule']]
    lines += ['',f"seed7达到预定三项标准的组：{', '.join(s['variant'] for s in qualified) if qualified else '无'}。",'',
              '### 同一入选方案的三训练种子复验','','|seed|参考ID/OOD|入选方案ID/OOD|ID变化%|OOD比变化%|最差类别变化%|','|---|---|---|---:|---:|---:|']
    chosen=selection['selected'];variants={7:f'taxibj_{chosen}_seed7',17:'taxibj_I62_seed17',27:'taxibj_I63_seed27'}
    paired=[]
    for seed,name in variants.items():
        r=records[references[seed]];e=records[name];i,o,w=score(e,r);paired.append(i<=1.03 and o<=.97 and w<=1.05)
        lines.append(f"|{seed}|{r['sets']['in_distribution']['metrics']['mae']:.5f}/{sum(category_metrics(r,'mae'))/3:.5f}|{e['sets']['in_distribution']['metrics']['mae']:.5f}/{sum(category_metrics(e,'mae'))/3:.5f}|{100*(i-1):+.3f}|{100*(o-1):+.3f}|{100*(w-1):+.3f}|")
    lines += ['',f'三个种子分别达标：{paired}。', '',
        '若只某个seed改善，应报告种子依赖；若全部未改善，不强行推荐新模型。不能用某一新组的seed7对照原模型seed17。三个训练种子仍不能自动支持显著性结论，mask复测不是训练种子。', '',
        '输入来源分离提供的是机制可检验性，不保证ID或OOD误差下降。移除显式预测补位并不意味着隐藏状态不含预测信息；H_i、差分和历史依旧保留。静态剩余预算只由可见观测构造，不是上一批的动态预测残差。', '',
        '下一步依据直接对照识别哪条输入路径的改变在同seed、三类OOD和无约束区域中持续有效；优先复验稳定的小改动。若新方法没有更好的折中，保留原参考并完整披露负结果，不新增测试后挑选的组合。']
    (assets/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    # Normalize spacing before tables and after headings for Markdown renderers.
    import re
    text='\n'.join(lines)+'\n';text=re.sub(r'(?m)^(#{1,6} .+)\n(?=\S)',r'\1\n\n',text)
    path.write_text(text);(suite/'report_path.txt').write_text(str(path)+'\n')
    return path
