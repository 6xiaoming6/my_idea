"""I29 F-only comparison; no modified baseline scores in the main table."""
import json,math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from run_b3_c3 import load,write,digest
from report_scale_delta_exploration import collect,fmt


def export_report(root,suite,js):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite)
    name,j=next(iter(js.items()));r,h,e,issues=collect(suite,name,j)
    protocol=j['config']['experiment_plan']['protocol']
    reference=load(suite/'references/I25/evaluations.json')
    if reference['protocol_sha256']!=digest(protocol):issues.append('I25 protocol mismatch')
    scores={'I25（FMC参考）':{k:v['metrics'] for k,v in reference['sets'].items()}}
    for file in sorted((suite/'references/baselines').glob('*.json')):
        baseline=load(file)
        if baseline['status']!='finished' or set(baseline['sets'])!=set(protocol['evaluations']):issues.append('Incomplete baseline '+file.stem)
        scores[file.stem]={k:v['raw'] for k,v in baseline['sets'].items()}
        for k,v in baseline['sets'].items():
            if (v['effective_mask_seed']!=reference['sets'][k]['effective_mask_seed'] or v['raw']['count']!=reference['sets'][k]['metrics']['metric_missing_count']):issues.append('Baseline mask/count mismatch '+file.stem)
    if e.get('sets'):
        scores['I29（F-only）']={k:v['metrics'] for k,v in e['sets'].items()}
        for k,v in e['sets'].items():
            m=v['metrics']
            if m['metric_missing_count']!=reference['sets'][k]['metrics']['metric_missing_count']:issues.append('F metric count mismatch')
            if v['effective_mask_seed']!=reference['sets'][k]['effective_mask_seed']:issues.append('F mask seed mismatch')
            for key in ['v25_observed_fraction_M','v25_observed_fraction_C']:
                val=m.get('coe_'+key,m.get(key))
                if val is not None and val!=0:issues.append('Coarse observation exposure '+key)
    complete=r['complete'] and not issues
    phase='完整' if complete else '阶段'
    stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v25_I29严格F输入基线对照_'+suite.name
    dest=root/'experments_report';dest.mkdir(exist_ok=True)
    assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    write(assets/'audit.json',{'complete':complete,'issues':issues,'run':r})
    write(assets/'curves.json',h);write(assets/'evaluations.json',scores)
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}。套件 `{suite}`，训练seed7。',
      '', '## 1. 具体方法与复现配置','',
      '|实验|输入与具体方法|比较目的|','|---|---|---|',
      '|I29|I25骨干严格只读F；前三轮自由计算尺度末轮F、共享八专家原生Top2、受控差分、因果历史池；无M/C投影或辅助损失|与F-only传统基线比较|',
      '|I25（已完成参考）|原多尺度FMC融合、真实逐轮观测、学习投影及一致性损失|输入及依赖观测的机制整体比较，非单模块消融|',
      '|ChannelMean / NodeMean|训练可见F通道均值 / 网格位置均值|无新增模块的统计参考|',
      '|TemporalLinear|窗口内时间插值，整段缺失回退NodeMean|时间关联参考|',
      '|SpatialMean|可见邻域均值，验证选3/5/9窗口|空间关联参考|',
      '|SoftImpute|12×2048窗口矩阵软阈值SVD，NodeMean中心化|低秩参考|','',
      'TaxiBJ两通道、T=12、四族混合mask0.4、seed7、100epoch、batch32、每5轮验证、AdamW、1e-3余弦至3e-4、weight_decay1e-4、clip1、AMP。只按ID验证MAE选best，六套协议与F掩码沿用I25。GPU1，GPU日志15秒采样；best/last保存完整训练状态以恢复。',
      'data.input_scales=F，训练/验证/测试M/C值及mask全部置零，不返回M/C完整标签；H0由原融合层的F槽形成。后续C/M仅为F隐藏状态的内部计算粒度，不能称为额外物理尺度观测。每窗口4轮×Top2共8次专家调用。历史池逐轮读取0/1/2/3条由可见F回填并detach的历史预测。',
      '移除学习分配投影头、可见M/C一致性辅助损失及无效的前后F损失混合标量。仅F缺失位置L1＋0.01候选均衡；无额外非负截断。剩余参数逐项核对与I25公共未训练初值一致，独立训练，不从I25训练检查点微调；参数量减少如实报告。',
      '传统基线只采用历史结果raw字段，不使用+P。旧RITS/BRITS训练曾增加额外L1，暂不纳入合规对比；本组不重新训练它们。分块、中心化和回退规则属于公开列明的适配配置，并非论文最佳配置复现。',
      f'复现：`CUDA_VISIBLE_DEVICES=1 python {suite}/source_snapshot/scripts/v25/run_taxibj_input_communication.py --suite {suite}`。配置见jobs/configs，原始数据及源码哈希见plan.json，公共初值一致性见i25_initialization_parity.json，GPU预检见acceptance/preflight.json。',
      '', '## 2. 完整结果与诊断','',
      '|组|状态|epoch/best|验证MAE|训练分钟|总/可训参数|峰值GiB|','|---|---|---|---|---|---|---|',
      f"|I29|{r['state']}|{r['epochs']}/{r['best_epoch']}|{fmt(r['val_mae'])}|{fmt(r['minutes'])}|{r['params']}/{r['trainable']}|{fmt(r['peak_gib'])}|",'',
      '|方法|协议|MAE|RMSE|缺失元素数|','|---|---|---|---|---|']
    for method,sets in scores.items():
        for label,v in sets.items():lines.append(f"|{method}|{label}|{fmt(v['mae'])}|{fmt(v['rmse'])}|{int(v.get('count',v.get('metric_missing_count',0)))}|")
    lines+=['','### I29训练曲线与过程诊断','', '|epoch|训练MAE|验证MAE|','|---|---|---|']
    for row in h:
        if row.get('val'):lines.append(f"|{row['epoch']}|{fmt(row['train']['mae'])}|{fmt(row['val']['mae'])}|")
    if h:
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot([v['epoch'] for v in h],[v['train']['mae'] for v in h],label='train')
        vals=[v for v in h if v.get('val')];ax.plot([v['epoch'] for v in vals],[v['val']['mae'] for v in vals],label='ID val')
        ax.set(xlabel='Epoch',ylabel='MAE');ax.legend();fig.tight_layout();fig.savefig(assets/'curves.png',dpi=150);plt.close(fig)
        lines+=['',f'![训练曲线]({assets.name}/curves.png)']
    if e:
        write(assets/'i29_full_diagnostics.json',e)
        lines+=['',f'[六协议全部门控、尺度路径、历史与分区诊断]({assets.name}/i29_full_diagnostics.json)。无粗尺度约束区域的计数应为0，相关空区域指标显示“—”，不记为零误差。']
    if issues:lines+=['','核验问题：'+str(issues)]
    if r['failure']:lines+=['','失败记录：'+str(r['failure'])]
    lines+=['','## 3. 分析与边界','']
    if complete:
        groups={'ID':['in_distribution'],'二元组合':['unseen_combinations','unseen_combinations_repeat'],'未见形态':['unseen_geometry','unseen_geometry_repeat'],'三元组合':['unseen_triple']}
        lines+=['以下为I29相对时间线性插值的误差变化，负数为改善；二元和形态先分别平均两次mask测试。','','|类别|MAE变化|RMSE变化|','|---|---|---|']
        for cat,keys in groups.items():
            diff=[(sum(scores['I29（F-only）'][k][metric] for k in keys)/sum(scores['TemporalLinear'][k][metric] for k in keys)-1)*100 for metric in ['mae','rmse']]
            lines.append(f'|{cat}|{diff[0]:+.2f}%|{diff[1]:+.2f}%|')
    else:lines+=['训练或评估尚未完成，不给最终效果结论。']
    lines+=['','与F-only基线相比，本组消除了额外M/C观测输入差异；仍有模型容量、训练和全局上下文差异，不能由此独立证明差分或历史池有效。I29与I25的差异同时涉及信息及其依赖机制，不能解释为仅改变H0或投影的单因素实验。',
      '不保证逐轮误差下降。单训练seed及mask复测不构成统计显著性证据。强神经网络对照尚缺修正后的原生结果；不能宣称优于所有补全方法。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'report_path.json',{'path':str(path),'complete':complete,'issues':issues})
    return path
