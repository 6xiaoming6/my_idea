"""Formal final seed7 report: methods/configuration, complete evidence, analysis."""
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from run_b3_c3 import load
from final_input_config import METHODS, DEFINITIONS, category_metrics


def score(record, reference):
    ratios=[a/b for a,b in zip(category_metrics(record,'mae'),category_metrics(reference,'mae'))]
    id_ratio=record['sets']['in_distribution']['metrics']['mae']/reference['sets']['in_distribution']['metrics']['mae']
    return id_ratio,sum(ratios)/3,max(ratios)


def export_report(suite,payload,jobs,selection):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib_final_input')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    root=Path(payload['project_root'])
    path=root/'experments_report'/f'{date}_v25_I64-I75最后一轮seed7输入优化_完整分析报告.md'
    assets=path.parent/(path.stem+'_assets'); assets.mkdir(parents=True,exist_ok=True)
    names=[r['variant'] for r in payload['references'].values()]+list(jobs)
    groups={g:{n:load(suite/g/f'{n}.json') for n in names} for g in ['development','evaluations','confirmation']}
    for group,records in groups.items():
        for n,r in records.items():
            if r['status']!='finished' or len(r['sets'])!=6 or r['split']!=('val' if group=='development' else 'test'):
                raise RuntimeError('Incomplete formal evaluation '+group+'/'+n)
    records=groups['evaluations']; reference=records['taxibj_I37_seed7']
    lines=[f'# {date}_v25_I64-I75最后一轮seed7输入优化_完整分析报告','',f'冻结套件：`{suite}`。十二次独立训练，全部seed7；I37/I56/I57为只读参考。','',
        '## 1. 实验具体做法与复现配置','',
        '|组|直接对照|实际做法|所检验问题|support/position/预测gain|','|---|---|---|---|---|']
    for n,(parent,description) in METHODS.items():
        s=DEFINITIONS[n]
        lines.append(f"|{n}|{parent}|{description}|{s['question']}|{s['support_gain']}/{s['position_gain']}/{s['prediction_gain']}|")
    lines += ['', '### 公共模型、训练与输入定义','',
        'TaxiBJ沿用2452/342/707窗口；双通道12时间步，F/M/C空间32/16/8，三尺度真实观测各自缺失，训练为四类单缺失族混合0.4。H0三尺度编码融合、四轮共享八专家，每轮原生Top2；前三轮自由尺度、第四轮F；原受控差分、历史池、学习分配投影保持I37。','',
        '100epoch、batch32、每5轮ID验证MAE选best，GPU1单卡AMP串行；AdamW1e-3余弦至3e-4，wd1e-4，grad_clip_norm1。前后缺失F损失各0.5，可见rawF0.1，可见M/C一致性0.1，候选均衡0.01。均从相同未训练初始化独立训练，未加载已训练参考权重。','',
        'support_gain和position_gain仅缩放逐轮专家输入，H0和路由元数据保持；native mask始终不变。旧整块zero补位会同时删除缺失M/C槽中可见F的聚合贡献，因此不能简单称为仅移除预测。I71–I75先构造where(m_F,X_F,g×P0)，再进行原空间mean聚合；原生可见M/C优先输入其真实观测密度。仅P0缺失槽被缩放，可见F始终保留，派生值不成为新观测。P0保持初始预测、不detach。','',
        '手算例：2×2区域可见F为10、40，缺失P0为8、12，则g=0/0.25/0.5/1时fallback为12.5/13.75/15/17.5；g=1与原补位精确一致。I71(g=0)与I57整块zero有本质区别。路由observed选项仅修改显式值统计为可见F均值、预测统计置零；隐藏状态中仍含模型估计，不能说路由只接触观测。','',
        '原六套测试协议保持；开发mask基种子2026102301–06，验证窗口；新确认mask2026102311–16，测试窗口，保留原split偏移。二元/形态各平均两次mask，再与三元类别等权；新mask不是新数据集或新训练种子。既有测试结果参与过方案设计，因此不是完全盲测。','',
        '先完成十二组开发评估并冻结selection.json，再开放本批测试。选择仅使用验证协议：ID≤I37同协议103%，按三类等权相对OOD MAE比值排序，同分依次OOD RMSE、ID、参数、编号。无合格组不强选新模型，不新增训练。最终推荐另要求OOD至少改善3%、最差类别退步≤5%。','',
        '原I37/I56/I57分别作为原模型、清理预测路径、再清理metadata的参考。I64–I66与I37构成support×position对照；I67/I68与I56/I57构成清理预测路径后的同类对照；I66/I69/I70/I57构成补位×路由对照；I71/I73/I74/I75构成预测强度×metadata强度对照，I72补充0.25预测剂量。','',
        f"冻结源码/设计/配置和数据摘要见[plan.json]({suite/'plan.json'})，公共初始化核验见[initialization_audit.json]({suite/'initialization_audit.json'})，CPU/GPU记录见[acceptance]({suite/'acceptance'})。",
        f"恢复入口：`python {suite/'source_snapshot/scripts/v25/run_final_input.py'} --suite {suite}`。",'',
        '|组|冻结作业|原始run|','|---|---|---|']
    receipts={n:load(suite/'results'/f'{n}.json') for n in jobs}
    for n,j in jobs.items(): lines.append(f"|{j['method']}|[配置]({suite/'jobs'/f'{n}.json'})|[日志/检查点]({receipts[n]['run_dir']})|")
    lines += ['','## 2. 完整结果、训练曲线与诊断','',
        '|模型|ID MAE|ID RMSE|二元MAE|形态MAE|三元MAE|三类OOD均值|ID变化%|OOD类别比变化%|最差类变化%|达标|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    summary=[]
    for n,r in records.items():
        i,o,w=score(r,reference); c=category_metrics(r,'mae'); m=r['sets']['in_distribution']['metrics']
        passed=i<=1.03 and o<=.97 and w<=1.05
        lines.append(f"|{n}|{m['mae']:.6f}|{m['rmse']:.6f}|{c[0]:.6f}|{c[1]:.6f}|{c[2]:.6f}|{sum(c)/3:.6f}|{100*(i-1):+.3f}|{100*(o-1):+.3f}|{100*(w-1):+.3f}|{'参考' if n=='taxibj_I37_seed7' else '是' if passed else '否'}|")
        summary.append(dict(variant=n,id_mae=m['mae'],id_ratio=i,ood_ratio=o,worst_ratio=w,meets_rule=passed))
    lines += ['','变化均相对同seed的原I37。OOD均值为三类别MAE平均，相对变化为三类别比值平均，两者分母不同。','']
    for g,title in [('evaluations','原六套完整测试'),('development','六套验证开发协议'),('confirmation','新mask确认协议')]:
        lines += [f'### {title}','','|模型|协议|MAE|RMSE|','|---|---|---:|---:|']
        for n,r in groups[g].items():
            for protocol,row in r['sets'].items(): lines.append(f"|{n}|{protocol}|{row['metrics']['mae']:.6f}|{row['metrics']['rmse']:.6f}|")
    lines += ['','### 训练完成情况、成本与曲线','',
        '|组|完成/best epoch|best验证MAE|参数/可训练参数|训练小时|峰值GiB|best/last|GPU日志|',
        '|---|---|---:|---|---:|---:|---|---|']
    curves=[]
    for n,r in receipts.items():
        run=Path(r['run_dir']); meta=load(run/'training_metadata.json')
        hist=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines() if s.strip()]
        if r['completed_epochs']!=100 or len(hist)!=100: raise RuntimeError('Incomplete training '+n)
        if not all((run/'checkpoints'/c).exists() for c in ['best.pth','last.pth']): raise RuntimeError('Missing checkpoint '+n)
        lines.append(f"|{n}|100/{r['best_epoch']}|{r['best_val_mae']:.6f}|{meta['total_params']}/{meta['trainable_params']}|{r['total_time_sec']/3600:.3f}|{max(s['perf']['peak_memory_gb'] for s in hist):.3f}|有|{'有' if (run/'logs/gpu_status.jsonl').exists() else '缺失'}|")
        fig,ax=plt.subplots(1,2,figsize=(9,3)); val=[s for s in hist if s.get('val')]
        ax[0].plot([s['epoch'] for s in hist],[s['train']['loss'] for s in hist]); ax[0].set_title('Train loss')
        ax[1].plot([s['epoch'] for s in val],[s['val']['mae'] for s in val]); ax[1].set_title('ID validation MAE')
        for a in ax: a.set_xlabel('epoch'); a.grid(alpha=.15)
        fig.tight_layout(); fig.savefig(assets/f'{n}.png',dpi=160); plt.close(fig)
        curves += [f"{n}：loss {hist[0]['train']['loss']:.4f}→{hist[-1]['train']['loss']:.4f}；末轮ID验证MAE {val[-1]['val']['mae']:.4f}，best在{r['best_epoch']}轮。",'',f'![{n}]({assets.name}/{n}.png)','']
    lines += ['', '置零输入通道仍保留并计算对应网络参数，参数相同不代表有效容量或梯度参与相同；不据此宣称FLOPs减少。峰值显存及耗时受实际路由影响。','',*curves,
        '### 输入及投影前后诊断','',
        '|模型|协议|投影前F MAE|投影后F MAE|约束区域MAE/计数|无约束区域MAE/计数|末轮fallback幅度|',
        '|---|---|---:|---:|---|---|---|']
    for n,r in records.items():
        for protocol in ['in_distribution','unseen_geometry','unseen_triple']:
            m=r['sets'][protocol]['metrics']
            def area(which):
                count=m.get('combination_post_'+which+'_count',0)
                return f"{m['combination_post_'+which+'_mae']:.5f}/{int(count)}" if count else '—/0'
            lines.append(f"|{n}|{protocol}|{m['combination_raw_all_mae']:.5f}|{m['mae']:.5f}|{area('constrained')}|{area('unconstrained')}|{m.get('coe_input_step4_estimate_abs','—')}|")
    lines += ['', '### 补位来源与重复元数据分量','',
        '|模型|协议|轮次|可见F补位贡献幅度|P0补位贡献幅度|support RMS|position RMS|',
        '|---|---|---:|---:|---:|---:|---:|']
    for n in jobs:
        for protocol in ['in_distribution','unseen_geometry','unseen_triple']:
            m=records[n]['sets'][protocol]['metrics']
            for step in range(1,5):
                keys=[f'coe_input_step{step}_{s}' for s in ['fallback_known_f_abs','fallback_prediction_abs','support_rms','position_rms']]
                values=[f'{m[k]:.6f}' if k in m else '—' for k in keys]
                lines.append(f"|{n}|{protocol}|{step}|{'|'.join(values)}|")
    (assets/'complete_diagnostics.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
    (assets/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines += ['',f'每轮fallback、有效观测计数、尺度路径、专家使用、历史/差分强度和投影残差完整保留于[诊断JSON]({assets.name}/complete_diagnostics.json)。fallback统计包含可见F的聚合贡献，不能统一理解为纯预测幅度。投影前指标读取raw预测，不使用已被覆盖的末轮输出。','',
        '### 冻结开发选择记录','','```json',json.dumps(selection,ensure_ascii=False,indent=2),'```','',
        '## 3. 分析、证据边界与建议','',
        '### 直接对照','', '|新组|直接对照|ID变化%|OOD类别比变化%|最差类变化%|','|---|---|---:|---:|---:|']
    parents={'I64':['I37'],'I65':['I37'],'I66':['I37','I64','I65'],'I67':['I57','I56'],
             'I68':['I57','I56'],'I69':['I66','I57'],'I70':['I66','I57'],'I71':['I57','I70'],
             'I72':['I71','I70'],'I73':['I71','I70'],'I74':['I71'],'I75':['I73','I74']}
    for n,controls in parents.items():
        for parent in controls:
            i,o,w=score(records[f'taxibj_{n}_seed7'],records[f'taxibj_{parent}_seed7'])
            lines.append(f'|{n}|{parent}|{100*(i-1):+.3f}|{100*(o-1):+.3f}|{100*(w-1):+.3f}|')
    chosen=selection['selected']
    passing=[s['variant'] for s in summary if s['variant'] in jobs and s['meets_rule']]
    lines += ['',f"开发协议ID合格并入选的新方案：{chosen or '无'}。测试达到三项数值标准的新组：{', '.join(passing) or '无'}。测试表用于完整披露，不反向改变开发选择。",'']
    if chosen:
        name=f'taxibj_{chosen}_seed7'; ci,co,cw=score(groups['confirmation'][name],groups['confirmation']['taxibj_I37_seed7'])
        selected_row=next(s for s in summary if s['variant']==name)
        lines += [f'开发入选方案在确认mask上：ID变化{100*(ci-1):+.3f}%，OOD类别比变化{100*(co-1):+.3f}%，最差类别变化{100*(cw-1):+.3f}%。','',
                  '开发入选方案达到原测试数值标准，可作为后续固定候选；仍需结合确认mask和机制对照判断。' if selected_row['meets_rule'] else '开发入选方案未达到原测试综合标准，不以开发排序包装为成功优化，保留已有参考。','']
    else: lines += ['本轮没有开发ID合格的新候选，按预定规则保留已有参考并结束优化，不启动额外组合。','']
    lines += ['本轮仅一个训练种子，不能宣称统计显著性或全局最优。确认mask复测不等于跨种子或跨数据集验证。每轮状态变化及观测校正不保证缺失位置误差逐轮下降。','',
        '对support/position的结论应从两组2×2对照共同判断；若效果依赖预测补位和路由设置，只能作条件性结论。I71与I57比较已知F证据恢复，I71/I72/I73/I70比较预测先验强度，I71/I73/I74/I75比较预定弱注入组合；不能把预测变化和已知观测删除混为一谈。','',
        '这些输入路径控制属于机制诊断及工程优化，清零、固定缩放或常规池化本身不是新的通用算法。改动是否有用，以ID/OOD折中、确认mask和对应控制为依据，不把平均指标改善直接当成理论解释已成立。','',
        '下一步按冻结规则保留候选或旧参考，转入方法梳理和论文验证；本批不扩展第十三组训练，也不新增训练种子。']
    import re
    text='\n'.join(lines)+'\n'; text=re.sub(r'(?m)^(#{1,6} .+)\n(?=\S)',r'\1\n\n',text)
    path.write_text(text); (suite/'report_path.txt').write_text(str(path)+'\n')
    return path
