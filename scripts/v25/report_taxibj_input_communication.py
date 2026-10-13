#!/usr/bin/env python3
"""Evidence-checked selected I01-I05 report; no training or checkpoint mutation."""
import csv,json,math,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from report_scale_delta_exploration import collect,fmt


def export_report(root,suite,js):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite);records=[];histories={};evals={};issues=[];sources=[]
    allowed={f'taxibj_I{i:02d}_seed7' for i in range(1,6)}
    if not js or not set(js).issubset(allowed):raise ValueError('Expected a nonempty subset of I01-I05')
    methods=[j['method'] for j in js.values()]
    if len(set(methods))!=len(methods):issues.append('Duplicate configured method')
    protocol=next(iter(js.values()))['config']['experiment_plan']['protocol']
    for n,j in js.items():
        row,h,e,bad=collect(suite,n,j);records.append(row);histories[j['method']]=h;evals[j['method']]=e;issues+=bad
        if j['config']['experiment_plan']['protocol']!=protocol:issues.append(n+': protocol mismatch')
        sources.append((j['method'],str(suite/'configs'/f'{n}.json'),row['run']))
    for method in ['A2','B1','B2']:
        directory=suite/'references'/method;c=load(directory/'config.json');r=load(directory/'receipt.json');e=load(directory/'evaluation.json');meta=load(directory/'metadata.json')
        h=[json.loads(x) for x in (directory/'history.json').read_text().splitlines() if x.strip()]
        valid=(r['status']==e['status']=='finished' and r['completed_epochs']==100 and
               r['config_sha256']==e['config_sha256']==digest(c) and e['protocol_sha256']==digest(protocol) and set(e['sets'])==set(protocol['evaluations']))
        if not valid:issues.append('Invalid frozen reference '+method)
        records.append({'name':'reference_'+method,'method':method,'state':'冻结参考','complete':valid,'epochs':r['completed_epochs'],'best_epoch':r['best_epoch'],'val_mae':r['best_val_mae'],'minutes':r['total_time_sec']/60,'params':meta['total_params'],'trainable':meta['trainable_params'],'peak_gib':max(x.get('perf',{}).get('peak_memory_gb',0) for x in h),'failure':None})
        histories[method]=h;evals[method]=e;sources.append((method,str(directory/'config.json'),r['run_dir']))
    results=[];families=[]
    for method,e in evals.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics']
            if not all(isinstance(m.get(k),(int,float)) and math.isfinite(m[k]) for k in ['mae','rmse']):issues.append(method+': nonfinite metrics '+label)
            results.append({'method':method,'protocol':label,'mae':m['mae'],'rmse':m['rmse'],'samples':ev['samples'],'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank'),'expert_calls':m.get('coe_expert_execution_count'),'area_proxy':m.get('coe_expert_grid_equivalents')})
            for k,v in m.items():
                if k.startswith('coe_family_') and k.endswith('_mae'):families.append({'method':method,'protocol':label,'family':k[len('coe_family_'):-4],'mae':v,'rmse':m.get(k[:-3]+'rmse'),'count':m.get(k[:-3]+'count')})
    final=bool(js) and all(r['complete'] for r in records) and not issues;phase='完整' if final else '阶段'
    stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v25_TaxiBJ输入通信与一致性验证_'+suite.name
    dest=root/'experments_report';dest.mkdir(parents=True,exist_ok=True);assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for name,value in [('summary.json',records),('curves.json',histories),('evaluations.json',evals),('audit.json',{'complete':final,'issues':issues})]:write(assets/name,value)
    for name,rows in [('protocols.csv',results),('families.csv',families)]:
        if rows:
            with (assets/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for name,h in histories.items():
        if h:
            axes[0].plot([x['epoch'] for x in h],[x['train']['mae'] for x in h],label=name)
            vals=[x for x in h if x.get('val')];axes[1].plot([x['epoch'] for x in vals],[x['val']['mae'] for x in vals],label=name)
    for ax,title in zip(axes,['Train MAE','ID validation MAE']):ax.set(title=title,xlabel='Epoch',ylabel='MAE');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=160);plt.close(fig)
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}；本套件{len(js)}组（{", ".join(methods)}），训练seed7；套件 `{suite}`。','',
        '## 1. 实验具体做法与复现信息','',
        '|组|初始H0|尺度与通信|参考与问题|','|---|---|---|---|',
        *[row for method,row in {
            'I01':'|I01|仅F，原融合层M/C槽置零|CMFF＋完整差分；各轮注入对应原生观测|B2：改变初始多尺度注入时机；不是全程仅F|',
            'I02':'|I02|F/M/C融合|四轮自由硬选单尺度、共享Top2＋完整差分|B2：自由尺度与差分的组合|',
            'I03':'|I03|F/M/C融合|CMFF＋完整差分＋自适应预测历史池|B2：历史读取的增量作用；不含乘积交互|',
            'I04':'|I04|F/M/C融合|CMFF＋完整差分＋最终F聚合对完整M/C真值的辅助L1|B2：全目标粗尺度监督基线；系数0.1；无历史池|',
            'I05':'|I05|F/M/C融合|CMFF＋完整差分＋观测驱动的剩余区域加权非负校正|B2：仅可见观测约束；新增分配头；无历史池/辅助粗损失|',
        }.items() if method in methods],
        '|旧A2|仅F，M/C全程不可用|CMFF＋完整差分|辅助参考：与I01的区别是各轮是否还能读M/C|',
        '|旧B1|F/M/C融合|CMFF，无差分|辅助无差分参考|',
        '|旧B2|F/M/C融合|CMFF＋完整差分|各新增组的主要参考|','',
        '### 共同协议、状态流与实现','',
        '复用TaxiBJ旧B2清理数据及四族混合mask0.4（不是纯random点缺失），seed7、100epoch、batch32、val每5轮、AdamW lr1e-3余弦到3e-4、weight_decay1e-4、clip1、AMP与严格确定性。仅ID验证MAE选best，last完整保存优化器/调度器/scaler/RNG，共同基础损失为F缺失位置L1＋0.01候选均衡；I04另加完整M/C目标监督，I05不增加该损失。',
        'F/M/C分别为[B,2,12,32,32]、[B,2,12,16,16]、[B,2,12,8,8]；完整F先按区域求和构造M/C，各尺度独立生成观测mask，编码时M/C除已知面积4/16。模型读取可见值，隐藏占位被屏蔽；目标只用于损失/评估。最终补全保持F形状并保留已观测F。',
        'H0之后四轮共享T/S/TD/SD/TA/ST/TL/SL八专家，每轮只执行原生Top2，共8次。CMFF是固定计算尺度与观测读取顺序，动态的是专家；I02才动态选尺度。各轮输出恢复F隐藏网格，首轮无历史差分，其余三轮以H当前−H前轮经detach比例RMS匹配、±0.1条件门修正专家输入；当轮路由读未修正状态。无中间轮监督，不能声称逐轮误差必降。',
        'I01保留原1×1融合投影，用[EF,0,0]得到H0，而非移除投影。dataset仍提供F/M/C；C在第一轮、M在第二轮进入模型状态，之后两轮注入F。公共参数/初始化与B2一致，初始不编码M/C因此计算略变，M/C投影槽无数据梯度，但这些尺度的原始观测在对应轮仍有梯度。',
        'I02使用现有S01硬尺度选择和选中分支straight-through梯度，无软预热。每轮每样本只执行选中的一个尺度，H0仍含所有尺度；不是只有被选尺度才首次可见。',
        'I03复用X05 HistoryReader，自适应逐位置读取绝对预测历史，不加入Y12乘积差值分支。P每个forward初始为空，四轮读取0/1/2/3条；每轮写入以可见F覆盖的F网格预测并detach。评分输入为归一化旧预测、最近预测、绝对差、年龄、可见覆盖；softmax在历史条目间归一化。读取后经共享零初始化MLP、tanh与0.1×当前隐藏RMS限制，仅在F缺失区域修正专家输入；加在差分修正之后。池不跨batch、不读当前未来或真值。',
        'I04以可见F回填最终预测，然后按与数据一致的2×2/4×4求和得到M/C；各自对预先构造的完整x_m_gt/x_c_gt计算全位置L1，分别除面积4/16后取两尺度均值，乘0.1加入基础损失。完整M/C目标只进入损失，不进入模型前向；不以M/C观测mask筛掉目标。测试时不执行真值校正。',
        'I05在最终F预测后，将可见M约束先减去区域内可见F，再从可见C总量扣除已观测M整块总量以及剩余区域内可见F，得到互不重叠的剩余未知区域；只在这些区域执行非负加权单纯形投影。共享轻量权重头读取最终隐藏状态、归一化F预测及F/M/C可见mask，权重有界于exp(-2)至exp(2)，使用任务损失学习校正分配；末层零初始化给出均匀权重投影，初始输出不保证等于B2。可见F保持原值，无粗尺度可见约束的未知位置保留原预测；不读取完整M/C目标、不增加粗尺度辅助损失。',
        'I04和I05均无历史池；它们不是在I03上叠加。二者监督范围不同：I04使用所有M/C训练目标（虽由已用F真值确定性聚合，仍改变监督目标及权重），I05只使用模型可见观测作为推理约束。二者差异不能解释成同监督下纯软约束与硬约束的因果对照。',
        '本套件所选各组均从头训练，公共初值逐张量与旧B2核对；I03历史模块、I05一致性权重头在独立fork RNG下新增，披露新增容量，不声称与B2等参数。I04不增加可训练参数。无FFN、三平面或额外视图；I04/I05各自只加入下述一致性干预。',
        '本套件所选组物理GPU1串行；各run日志含每15秒GPU状态，未启用周期线程栈采样。旧B2存在历史中断/换卡记录，比较数值协议一致不代表执行历程完全相同。',
        f'六套协议：`{json.dumps(protocol,ensure_ascii=False)}`。训练四族逐epoch重采样，val/test固定且分别偏移基础mask seed 20000/30000；测试mask复测不是独立训练种子。','',
        '### 数据来源与复现','',f'计划 `{suite/"plan.json"}`；源码 `{suite/"source_snapshot"}`；初始化 `{suite/"initialization_audit.json"}`；预检 `{suite/"acceptance/preflight.json"}`。',
        f'恢复：`python -u {suite/"source_snapshot/scripts/v25/run_taxibj_input_communication.py"} --suite {suite}`，外部设置CUDA_VISIBLE_DEVICES=1和OMP/MKL/OPENBLAS线程2；完成组跳过，完整队列结束后tmux退出。','',
        '|组|配置|run|','|---|---|---|']
    lines += [f'|{m}|`{c}`|`{r}`|' for m,c,r in sources]
    lines += ['','## 2. 实验结果','','|组|状态|epoch/best|val MAE|累计训练分钟|总/可训参数|峰值GiB|','|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["state"]}|{r["epochs"]}/{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in records]
    lines += ['','|组|协议|MAE|RMSE|ms/窗口|专家执行数|面积代理|','|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{fmt(r["expert_calls"])}|{fmt(r["area_proxy"])}|' for r in results]
    lines += ['','### 分缺失族结果','','|组|协议|族|MAE|RMSE|计数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    lines += ['','### 训练曲线与机制诊断','',f'![曲线]({assets.name}/curves.png)','',f'[全部epoch与曲线]({assets.name}/curves.json)；[全部路由、门、RMS、历史、逐轮诊断]({assets.name}/evaluations.json)。','',
        '|组|首/末train MAE|最优/末val MAE|','|---|---|---|']
    for n,h in histories.items():
        if h:
            vals=[x['val']['mae'] for x in h if x.get('val')];lines.append(f'|{n}|{fmt(h[0]["train"]["mae"])}/{fmt(h[-1]["train"]["mae"])}|{fmt(min(vals) if vals else None)}/{fmt(vals[-1] if vals else None)}|')
    lines += ['','|组|协议|四轮读出MAE|四轮门绝对均值|四轮历史修正均值|','|---|---|---|---|---|']
    for method,e in evals.items():
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];vals=['/'.join(fmt(m.get(pattern.format(i=i))) for i in range(1,5)) for pattern in ['coe_x_step{i}_mae','coe_step{i}_memory_gate_abs_mean','coe_x_step{i}_pool_delta_abs']];lines.append('|'+method+'|'+label+'|'+'|'.join(vals)+'|')
    if set(methods)&{'I04','I05'}:
        lines+=['','### 一致性监督与输出约束诊断','','|组|协议|M/C聚合观测残差|投影前M/C残差|约束覆盖比例|修正幅度|权重最小/最大|','|---|---|---|---|---|---|---|']
        for method in methods:
            for label,ev in evals[method].get('sets',{}).items():
                m=ev['metrics'];after='/'.join(fmt(m.get('coe_v25_observed_sum_residual_'+s)) for s in ('M','C'))
                before='/'.join(fmt(m.get('coe_v25_consistency_before_observed_sum_residual_'+s)) for s in ('M','C'))
                weights='/'.join(fmt(m.get('coe_v25_consistency_weight_'+k)) for k in ('min','max'))
                lines.append(f'|{method}|{label}|{after}|{before}|{fmt(m.get("coe_v25_consistency_constrained_fraction"))}|{fmt(m.get("coe_v25_consistency_correction_abs"))}|{weights}|')
        lines+=['','只有聚合残差下降不能证明细粒度MAE改善。I05的校正位于最终读出之后，不能将校正后输出与前几轮读出当作未经处理的逐轮修复轨迹。全量训练损失分量与约束组数见curves.json、evaluations.json；缺失诊断显示为“—”，不补造。']
    lines+=['','### 完成与失败审计','',('；'.join(issues) if issues else '已有凭据/配置/检查点/主指标未发现核验异常；未完成组以上表为准。')]
    for r in records:
        if not r['complete']:lines.append(f'- {r["method"]}：未完成。')
        if r['failure']:lines.append(f'- {r["method"]}历史失败：`{r["failure"]}`。')
    lines+=['','## 3. 结果分析与建议','','### 预设配对的实测变化','','负数表示误差改善；不省略负结果。','','|配对|协议|MAE变化%|RMSE变化%|','|---|---|---|---|']
    pairs=[(m,ref) for m in methods for ref in ('B2','B1')]
    if 'I01' in methods:pairs.append(('I01','A2'))
    if 'I04' in methods and 'I05' in methods:pairs.append(('I05','I04'))
    for a,b in pairs:
        ea,eb=evals.get(a,{}).get('sets',{}),evals[b].get('sets',{})
        for lab in protocol['evaluations']:
            if lab in ea and lab in eb:
                x,y=ea[lab]['metrics'],eb[lab]['metrics'];lines.append(f'|{a}/{b}|{lab}|{100*(x["mae"]/y["mae"]-1):+.3f}|{100*(x["rmse"]/y["rmse"]-1):+.3f}|')
    if final:
        for name in methods:
            changes={lab:100*(ev['metrics']['mae']/evals['B2']['sets'][lab]['metrics']['mae']-1) for lab,ev in evals[name]['sets'].items()};ood=[v for k,v in changes.items() if k!='in_distribution'];wins=sum(v<0 for v in ood)
            lines+=['',f'{name}相对B2：ID MAE变化{changes["in_distribution"]:+.2f}%；五套OOD中{wins}套MAE改善，变化范围{min(ood):+.2f}%至{max(ood):+.2f}%。']
    else:lines+=['','本批尚未完成；已出现的局部结果不代表本套件最终排序。']
    lines+=['','### 解释边界与下一步','',
        'I01/B2主要隔离初始入口，但同时改变初始状态、早期差分来源及初始数值后备；若改善，只支持改变入口有帮助，不足以证明原泛化退步由差分独立造成。仍需同入口无差分对照检验交互。I01/A2则包含逐轮可见M/C的额外信息，不能归为纯算法收益。',
        'I02/B2比较的是当前硬尺度估计器与完整差分的组合，若路径集中，需要固定该路径对照；没有对应同协议自由尺度无差分结果时，不能称为二者互补。不得以BikeNYC的S01结果替代TaxiBJ配对。',
        'I03/B2包含新增历史模块容量与绝对预测读取，不能单凭改善证明历史与差分不冗余；需要历史无差分、仅最近历史、等容量分支等后续控制。历史权重不是因果重要性，零初始化不保证训练后不破坏泛化。',
        'I04/B2检验完整粗尺度监督的增量；I05/B2检验仅可见观测约束下的学习校正。I05/I04同时改变监督范围、输出约束、计算及参数量，不能单独证明自适应分配优于同信息量软约束。仍需可见观测软损失、固定联合投影和等容量修正头。',
        'I05只处理本批精确、对齐、相容的嵌套非负观测。固定欧氏投影的平方误差不增性质不自动适用于学习权重，也不保证MAE不增。约束组分解属于实现机制，独立新颖性和稳定泛化须由后续控制支持。',
        '单seed7、同率训练测试，不支持显著性或稳定跨数据集结论；五套OOD及mask复测不是五个独立训练样本。只按ID验证选checkpoint，研究筛选仍已参考OOD，正式确认应保留新的条件/种子。中间轮无监督，逐轮读出不保证单调。MAPE受零值影响不作为主判据。',
        '所有成本是实际运行记录，未严格GPU独占；累计训练分钟不含故障等待与评估，不能等同墙钟时长。先逐协议检查已配置各组与B2和B1的改善/退步，再决定补机制对照，不根据单个最好结果直接更换主线。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n');write(suite/'report_path.json',{'path':str(path),'complete':final,'issues':issues});return path
