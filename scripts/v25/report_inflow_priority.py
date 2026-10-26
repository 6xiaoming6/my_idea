#!/usr/bin/env python3
"""I28 report: single-inflow evidence only, never rank against dual-variable MAE."""
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_after_suite import load, write, digest
from report_scale_delta_exploration import collect, fmt


def export_report(root,suite,jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite)
    if list(jobs)!=['taxibj_I28_seed7']:raise ValueError('Expected only I28')
    name,job=next(iter(jobs.items()));cfg=job['config']
    if cfg['model']['c_in']!=1 or cfg['data']['channel_selection']['source_index']!=0:
        raise ValueError('Expected single inflow')
    row,history,evaluation,issues=collect(suite,name,job)
    protocol=cfg['experiment_plan']['protocol']
    for label,ev in evaluation.get('sets',{}).items():
        spec=protocol['evaluations'][label]
        expected={'families':spec['families'],'rates':[spec.get('rate',protocol['rate'])],
                  'seed':spec['seed'],'resample_each_epoch':False}
        if ev.get('mask_spec')!=expected or ev.get('samples',0)<=0:issues.append('Mask/sample mismatch '+label)
    complete=bool(row['complete'] and not issues)
    phase='完整' if complete else '阶段'
    stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v25_I28单变量流入验证_'+suite.name
    dest=root/'experments_report';dest.mkdir(parents=True,exist_ok=True)
    assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for key,value in [('summary',row),('curves',history),('evaluations',evaluation),('audit',{'complete':complete,'issues':issues})]:
        write(assets/(key+'.json'),value)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for split in ('train','val'):
        points=[h for h in history if h.get(split)]
        for ax,metric in zip(axes,('mae','loss')):
            if points:ax.plot([h['epoch'] for h in points],[h[split][metric] for h in points],label=split)
            ax.set(xlabel='Epoch',ylabel=metric.upper());ax.grid(alpha=.25)
    if history:
        for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=150);plt.close(fig)
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}；单训练seed7；本套件仅I28，不混入双通道任务排名。','',
        '## 1. 具体做法与复现配置','',
        '|实验|基础与具体改动|验证问题|','|---|---|---|',
        '|I28|I25改为单变量流入，F/M/C同时提取原第0通道，c_in=1；从新随机初始化训练|单变量主线的可行性与ID/OOD表现|','',
        'F/M/C分别为[B,1,12,32,32]、[B,1,12,16,16]、[B,1,12,8,8]。样本顺序和划分沿用I25，聚合为逐时间、逐变量空间求和。数据切片发生在独立NPZ中；训练与推理不读取流出。掩码仍沿用四类混合0.4和独立尺度RNG，不是纯随机点缺失。',
        'H0融合FMC；前三轮自由硬选尺度、第四轮F；四轮共享T/S/TD/SD/TA/ST/TL/SL八专家，每轮原生Top2。RMS匹配±0.1受控差分只修正专家输入，路由读取未修正状态；历史池逐轮读取0/1/2/3条detach预测，回填可见F，不跨batch；末轮学习分配剩余区域投影保持可见F，满足可见M/C非负剩余总量约束。',
        '训练损失为(1−β)最终F缺失L1＋β原始F同位置L1＋0.1投影前可见M/C面积归一化L1＋0.01候选均衡；β为sigmoid全局标量、初始0.3。原始预测取独立字段，不使用已投影的第四轮读出。',
        f"seed{cfg['seed']}，{cfg['train']['epochs']}epoch，batch{cfg['data']['batch_size']}，每{cfg['train']['val_epoch']}轮验证；AdamW，lr1e-3余弦到3e-4，weight_decay1e-4，clip1，AMP及完整恢复；仅ID验证MAE选best。GPU1单卡串行，run内每15秒GPU日志。",
        '单通道从seed7随机初始化，不继承I25训练权重，也不与双通道初始化逐张量相等；参数尺寸和数量随输入输出通道变化。无中间轮监督，不保证逐轮误差下降。','',
        f"冻结配置：`{suite/'configs'/f'{name}.json'}`；运行目录：`{row['run']}`。",
        f"冻结源码与数据摘要：`{suite/'plan.json'}`；初始化：`{suite/'initialization_audit.json'}`。",
        f"入口：`python -u {suite/'source_snapshot/scripts/v25/run_inflow_priority.py'} --suite {suite}`；依赖I26完成，I27保持暂停等待，成功后自动释放。已完成组跳过，失败停队列。",'',
        '## 2. 完整结果、训练曲线与诊断','',
        '|状态|epoch/best|验证MAE|训练分钟|总/可训练参数|峰值GiB|','|---|---|---|---|---|---|',
        f"|{row['state']}|{row['epochs']}/{row['best_epoch']}|{fmt(row['val_mae'])}|{fmt(row['minutes'])}|{row['params']}/{row['trainable']}|{fmt(row['peak_gib'])}|",'',
        '|测试协议|MAE|RMSE|ms/窗口|专家调用数|','|---|---|---|---|---|']
    for label in protocol['evaluations']:
        m=evaluation.get('sets',{}).get(label,{}).get('metrics',{})
        lines.append('|'+label+'|'+'|'.join(fmt(m.get(k)) for k in ('mae','rmse','forward_ms_per_sample_per_rank','coe_expert_execution_count'))+'|')
    lines+=['','### 逐缺失族','', '|协议|缺失族|MAE|RMSE|有效数|','|---|---|---|---|---|']
    for label,ev in evaluation.get('sets',{}).items():
        m=ev['metrics']
        for k in sorted(m):
            if k.startswith('coe_family_') and k.endswith('_mae'):
                lines.append(f"|{label}|{k[len('coe_family_'):-4]}|{fmt(m[k])}|{fmt(m.get(k[:-3]+'rmse'))}|{fmt(m.get(k[:-3]+'count'))}|")
    lines+=['','### 曲线与前后监督','',f'![训练与验证曲线]({assets.name}/curves.png)','',
            '|epoch|训练MAE|验证MAE|训练β|验证β|可见M/C辅助损失|','|---|---|---|---|---|---|']
    for h in history:
        if h['epoch']==1 or h['epoch']%5==0:
            tr,va=h.get('train') or {},h.get('val') or {}
            lines.append(f"|{h['epoch']}|{fmt(tr.get('mae'))}|{fmt(va.get('mae'))}|{fmt(tr.get('f_loss_raw_weight'))}|{fmt(va.get('f_loss_raw_weight'))}|{fmt(tr.get('l_visible_consistency_aux'))}|")
    lines+=['','### 原始与投影后分区域误差','',
            '|协议|区域|有效缺失数|原始MAE/RMSE|最终MAE/RMSE|','|---|---|---|---|---|']
    for label,ev in evaluation.get('sets',{}).items():
        m=ev['metrics']
        for area in ('all','constrained','unconstrained'):
            count=m.get(f'combination_raw_{area}_count')
            values=['/'.join(fmt(m.get(f'combination_{stage}_{area}_{k}')) for k in ('mae','rmse')) if count else '—' for stage in ('raw','post')]
            lines.append(f"|{label}|{area}|{fmt(count)}|{values[0]}|{values[1]}|")
    lines+=['','### 约束、历史及路径诊断','',
            '|协议|M残差原始/最终|C残差原始/最终|历史注入2/3/4轮|主要尺度路径|','|---|---|---|---|---|']
    for label,ev in evaluation.get('sets',{}).items():
        m=ev['metrics']; residual=['/'.join(fmt(m.get(f'combination_{stage}_visible_{scale}_mae')) for stage in ('raw','post')) for scale in ('m','c')]
        injection='/'.join(fmt(m.get(f'coe_x_step{i}_pool_delta_abs')) for i in (2,3,4))
        paths=sorted([(k[len('coe_scale_path_'):-len('_fraction')],v) for k,v in m.items() if k.startswith('coe_scale_path_') and k.endswith('_fraction')],key=lambda v:-v[1])[:4]
        lines.append(f"|{label}|{residual[0]}|{residual[1]}|{injection}|"+', '.join(f'{p}:{v:.4f}' for p,v in paths)+'|')
    lines+=['',f'完整指标、门控、历史和路径：[evaluations.json]({assets.name}/evaluations.json)；逐轮训练：[curves.json]({assets.name}/curves.json)。','',
            '### 缺失与失败披露','',str(issues) if issues else '已有记录未发现配置、检查点和主指标异常；是否完成以上述状态为准。',
            f"历史失败记录：`{row['failure']}`。",'', '## 3. 分析、边界与后续建议','']
    if complete:
        sets=evaluation['sets'];idmae=sets['in_distribution']['metrics']['mae']
        lines.append(f'I28已完成全部训练与六套测试，流入ID MAE为{idmae:.4f}。以下类别均值仅描述本组不同缺失条件，不能解释为相对旧双通道模型的改善。')
        for category,labels in [('二元组合',['unseen_combinations','unseen_combinations_repeat']),('未见形态',['unseen_geometry','unseen_geometry_repeat']),('三元组合',['unseen_triple'])]:
            ma=sum(sets[p]['metrics']['mae'] for p in labels)/len(labels)
            rm=sum(sets[p]['metrics']['rmse'] for p in labels)/len(labels)
            lines.append(f'- {category}：MAE {ma:.4f}，RMSE类别均值 {rm:.4f}；MAE相对本组ID变化{100*(ma/idmae-1):+.2f}%。')
        vals=[h for h in history if h.get('val')]
        if vals:lines.append(f"\n验证MAE从首次{vals[0]['val']['mae']:.4f}变化到末次{vals[-1]['val']['mae']:.4f}，best在第{row['best_epoch']}轮；具体波动及是否反弹需结合上图，不以可学习加权后的总loss替代任务误差。")
    else:lines.append('本组尚未完成或存在审计问题，不能给出最终性能结论；保留阶段记录。')
    lines+=['','旧I25汇总指标包含流入和流出；I28仅预测流入，不直接比较两者数值，不应用旧I08双通道ID阈值8.8709。输入维度和初始化变化同样需要披露。若要验证单变量是否优于联合建模，后续需对旧I25按一致掩码仅评分流入，并补齐单变量基线。',
            '本次只验证单变量完整组合；单seed及掩码复测不能证明稳定泛化、独立模块贡献或统计显著性。约束满足不保证误差下降，误差与路由变化不能直接证明机制。后续优先补同任务基线、关键消融及训练种子复验。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(suite/'report_path.json',{'path':str(path),'complete':complete,'issues':issues})
    return path
