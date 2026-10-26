#!/usr/bin/env python3
"""Evidence-checked I01-I27 reports with read-only frozen comparison suites."""
import csv,json,math,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from report_scale_delta_exploration import collect,fmt
from prepare_data import sha


def comparison_jobs(suite,js):
    """Resolve explicit frozen dependencies without adding any training jobs."""
    entries=[(Path(suite),name,j,False) for name,j in js.items()]
    expected=next(iter(js.values()))['config']
    for directory,stamp in load(Path(suite)/'plan.json').get('comparison_suites',{}).items():
        previous=Path(directory);plan_path=previous/'plan.json'
        if sha(plan_path)!=stamp:raise RuntimeError('Comparison plan changed '+directory)
        plan=load(plan_path)
        resolved=load(previous/'resolved_jobs.json') if (previous/'resolved_jobs.json').exists() else plan['jobs']
        selected_methods = load(Path(suite)/'plan.json').get('comparison_methods', {}).get(directory)
        for name,original in plan['jobs'].items():
            if name in js:continue
            if selected_methods is not None and original['method'] not in selected_methods:continue
            j=resolved[name];cfg=j['config']
            if cfg!=original['config'] or j['sources']!=original['sources']:raise RuntimeError('Comparison config/data paths changed '+name)
            if (cfg['train']!=expected['train'] or cfg['seed']!=expected['seed']
                    or cfg['experiment_plan']['protocol']!=expected['experiment_plan']['protocol']):
                raise RuntimeError('Comparison protocol/budget mismatch '+name)
            entries.append((previous,name,j,True))
    if len({name for _,name,_,_ in entries})!=len(entries):
        raise RuntimeError('Duplicate comparison job')
    return sorted(entries,key=lambda row:row[2]['method'])


OOD_CATEGORIES = {
    '二元组合': ('unseen_combinations', 'unseen_combinations_repeat'),
    '未见形态': ('unseen_geometry', 'unseen_geometry_repeat'),
    '三元组合': ('unseen_triple',),
}


def combination_ranking(evals, threshold=8.8709):
    reference = evals.get('I08', {}).get('sets', {})
    required = {'in_distribution', *(p for group in OOD_CATEGORIES.values() for p in group)}
    if not required.issubset(reference):
        return []
    rows = []
    for method, evaluation in evals.items():
        sets = evaluation.get('sets', {})
        if not required.issubset(sets):
            continue
        if not all(math.isfinite(sets[p]['metrics'][k]) for p in required for k in ('mae','rmse')):
            continue
        row = {'method': method, 'id_mae': sets['in_distribution']['metrics']['mae']}
        row['eligible'] = row['id_mae'] <= threshold
        for metric in ('mae', 'rmse'):
            ratios = {category: sum(sets[p]['metrics'][metric] / reference[p]['metrics'][metric]
                                   for p in protocols) / len(protocols)
                      for category, protocols in OOD_CATEGORIES.items()}
            row[metric+'_categories'] = ratios
            row[metric+'_score'] = sum(ratios.values()) / 3
            row[metric+'_worst_change_pct'] = 100 * (max(ratios.values()) - 1)
        rows.append(row)
    return sorted(rows, key=lambda r: (not r['eligible'], r['mae_score'], r['rmse_score'], r['id_mae'], r['method']))


def export_report(root,suite,js):
    if js and all(j['method']=='I29' for j in js.values()):
        from report_f_only import export_report as export_f_only
        return export_f_only(root,suite,js)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,suite=Path(root),Path(suite);records=[];histories={};evals={};issues=[];sources=[]
    allowed={f'taxibj_I{i:02d}_seed7' for i in list(range(1,28))+list(range(30,42))}
    if not js or not set(js).issubset(allowed):raise ValueError('Expected registered I-series jobs')
    entries=comparison_jobs(suite,js)
    methods=[j['method'] for _,_,j,_ in entries]
    if len(set(methods))!=len(methods):issues.append('Duplicate configured method')
    protocol=next(iter(js.values()))['config']['experiment_plan']['protocol']
    for origin,n,j,external in entries:
        row,h,e,bad=collect(origin,n,j)
        if external:row['state']='关联冻结组：'+row['state']
        records.append(row);histories[j['method']]=h;evals[j['method']]=e;issues+=bad
        if j['config']['experiment_plan']['protocol']!=protocol:issues.append(n+': protocol mismatch')
        sources.append((j['method'],str(origin/'configs'/f'{n}.json'),row['run']))
    paused=suite/'references/I09'
    if set(methods)&{'I10','I11'} and paused.exists():
        r=load(paused/'paused.json');e=load(paused/'evaluation.json');c=load(paused/'config.json');meta=load(paused/'metadata.json')
        h=[json.loads(x) for x in (paused/'history.json').read_text().splitlines() if x.strip()]
        valid=(r['status']=='paused_by_user' and e['status']=='finished' and
               r['best_epoch']==e['best_epoch'] and r['checkpoint']==e['checkpoint'] and
               e['config_sha256']==digest(c) and e['protocol_sha256']==digest(protocol) and
               set(e['sets'])==set(protocol['evaluations']))
        if not valid:issues.append('Invalid paused I09 comparison')
        records.append({'name':'paused_I09','method':'I09','state':'用户暂停，best六套评估完成；预算不等',
            'complete':False,'comparison_ready':valid,'epochs':r['completed_epochs'],
            'best_epoch':r['best_epoch'],'val_mae':r['best_val_mae'],
            'minutes':sum(x.get('perf',{}).get('epoch_time_sec',0) for x in h)/60,
            'params':meta['total_params'],'trainable':meta['trainable_params'],
            'peak_gib':max((x.get('perf',{}).get('peak_memory_gb',0) for x in h),default=0),'failure':None})
        methods.append('I09');histories['I09']=h;evals['I09']=e
        sources.append(('I09暂停参考',str(paused/'config.json'),str(Path(r['checkpoint']).parents[1])))
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
    final=bool(js) and all(r['complete'] or r.get('comparison_ready',False) for r in records) and not issues;phase='完整' if final else '阶段'
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
    lines=[f'# {stem}_{phase}分析报告','',f'状态：{phase}；本套件训练{len(js)}组，关联既有冻结组{len(entries)-len(js)}组；合计展示{", ".join(methods)}，训练seed7；套件 `{suite}`。','',
        '## 1. 实验具体做法与复现信息','',
        '|组|初始H0|尺度与通信|参考与问题|','|---|---|---|---|',
        *[row for method,row in {
            'I01':'|I01|仅F，原融合层M/C槽置零|CMFF＋完整差分；各轮注入对应原生观测|B2：改变初始多尺度注入时机；不是全程仅F|',
            'I02':'|I02|F/M/C融合|四轮自由硬选单尺度、共享Top2＋完整差分|B2：自由尺度与差分的组合|',
            'I03':'|I03|F/M/C融合|CMFF＋完整差分＋自适应预测历史池|B2：历史读取的增量作用；不含乘积交互|',
            'I04':'|I04|F/M/C融合|CMFF＋完整差分＋最终F聚合对完整M/C真值的辅助L1|B2：全目标粗尺度监督基线；系数0.1；无历史池|',
            'I05':'|I05|F/M/C融合|CMFF＋完整差分＋观测驱动的剩余区域加权非负校正|B2：仅可见观测约束；新增分配头；无历史池/辅助粗损失|',
            'I06':'|I06|F/M/C融合|CMFF＋完整差分＋仅可见M/C的聚合L1|I04：标签范围；I05/I07：可见信息下的软约束；系数0.1|',
            'I07':'|I07|F/M/C融合|CMFF＋完整差分＋固定等权剩余区域投影|I05：去掉学习权重；无额外参数、无历史池|',
            'I08':'|I08|F/M/C融合|CMFF＋完整差分＋自适应预测历史池＋自适应投影|I03/I05/B2：历史读取与输出约束的2×2对照|',
            'I09':'|I09|F/M/C融合|自由硬选尺度＋共享Top2＋完整差分＋历史池＋学习投影|I08：只加入I02尺度训练；I02/I03/I05：组合收益与损失|',
            'I10':'|I10|F/M/C融合|前三轮自由硬选尺度，第四轮固定F；共享Top2＋完整差分＋历史池＋学习投影|I09：仅限制末轮尺度；从头100轮。I09主动暂停，预算不等|',
            'I11':'|I11|F/M/C融合|前三轮逐样本无放回选F/M/C各一次，第四轮固定F；其余沿用I10|I10：覆盖约束；I08：同尺度次数下固定顺序与动态顺序|',
            'I12':'|I12|F/M/C融合|固定CMFF＋共享Top2＋完整差分＋历史池＋学习分配投影；同I08|I08同配置同seed重复核验；不是新结构或独立训练种子|',
            **{m: '|'+m+'|F/M/C融合|'+description+'|'+('I13' if 17<=int(m[1:])<=22 else parent)+'|'
               for m,(parent,description) in __import__('run_taxibj_input_communication').COMBINATIONS.items()},
        }.items() if method in methods],
        '|旧A2|仅F，M/C全程不可用|CMFF＋完整差分|辅助参考：与I01的区别是各轮是否还能读M/C|',
        '|旧B1|F/M/C融合|CMFF，无差分|辅助无差分参考|',
        '|旧B2|F/M/C融合|CMFF＋完整差分|各新增组的主要参考|','',
        '### 共同协议、状态流与实现','',
        '复用TaxiBJ旧B2清理数据及四族混合mask0.4（不是纯random点缺失），seed7、100epoch、batch32、val每5轮、AdamW lr1e-3余弦到3e-4、weight_decay1e-4、clip1、AMP与严格确定性。仅ID验证MAE选best，last完整保存优化器/调度器/scaler/RNG，共同基础损失为F缺失位置L1＋0.01候选均衡；I04另加完整M/C目标监督，I06只增加可见M/C软损失；I05/I07/I08/I09不增加粗尺度损失。',
        'F/M/C分别为[B,2,12,32,32]、[B,2,12,16,16]、[B,2,12,8,8]；完整F先按区域求和构造M/C，各尺度独立生成观测mask，编码时M/C除已知面积4/16。模型读取可见值，隐藏占位被屏蔽；目标只用于损失/评估。最终补全保持F形状并保留已观测F。',
        '除I27采用四套独立专家池外，H0之后四轮共享T/S/TD/SD/TA/ST/TL/SL八专家，每轮只执行原生Top2，共8次。CMFF是固定计算尺度与观测读取顺序，动态的是专家；I02/I09动态选尺度。各轮输出恢复F隐藏网格，首轮无历史差分，其余三轮以H当前−H前轮经detach比例RMS匹配、±0.1条件门修正专家输入；当轮路由读未修正状态。无中间轮监督，不能声称逐轮误差必降。',
        'I01保留原1×1融合投影，用[EF,0,0]得到H0，而非移除投影。dataset仍提供F/M/C；C在第一轮、M在第二轮进入模型状态，之后两轮注入F。公共参数/初始化与B2一致，初始不编码M/C因此计算略变，M/C投影槽无数据梯度，但这些尺度的原始观测在对应轮仍有梯度。',
        'I02使用现有S01硬尺度选择和选中分支straight-through梯度，无软预热。每轮每样本只执行选中的一个尺度，H0仍含所有尺度；不是只有被选尺度才首次可见。',
        'I03复用X05 HistoryReader，自适应逐位置读取绝对预测历史，不加入Y12乘积差值分支。P每个forward初始为空，四轮读取0/1/2/3条；每轮写入以可见F覆盖的F网格预测并detach。评分输入为归一化旧预测、最近预测、绝对差、年龄、可见覆盖；softmax在历史条目间归一化。读取后经共享零初始化MLP、tanh与0.1×当前隐藏RMS限制，仅在F缺失区域修正专家输入；加在差分修正之后。池不跨batch、不读当前未来或真值。',
        'I04以可见F回填最终预测，然后按与数据一致的2×2/4×4求和得到M/C；各自对预先构造的完整x_m_gt/x_c_gt计算全位置L1，分别除面积4/16后取两尺度均值，乘0.1加入基础损失。完整M/C目标只进入损失，不进入模型前向；不以M/C观测mask筛掉目标。测试时不执行真值校正。',
        'I05在最终F预测后，将可见M约束先减去区域内可见F，再从可见C总量扣除已观测M整块总量以及剩余区域内可见F，得到互不重叠的剩余未知区域；只在这些区域执行非负加权单纯形投影。共享轻量权重头读取最终隐藏状态、归一化F预测及F/M/C可见mask，权重有界于exp(-2)至exp(2)，使用任务损失学习校正分配；末层零初始化给出均匀权重投影，初始输出不保证等于B2。可见F保持原值，无粗尺度可见约束的未知位置保留原预测；不读取完整M/C目标、不增加粗尺度辅助损失。',
        'I04和I05均无历史池；它们不是在I03上叠加。二者监督范围不同：I04使用所有M/C训练目标（虽由已用F真值确定性聚合，仍改变监督目标及权重），I05只使用模型可见观测作为推理约束。二者差异不能解释成同监督下纯软约束与硬约束的因果对照。',
        'I06沿用I04的最终F回填、2×2/4×4聚合及面积归一化，但只在m_m/m_c可见位置对x_m_obs/x_c_obs计算L1；各尺度按该尺度可见单元数平均，再固定取两尺度均值，空观测尺度贡献0。系数仍0.1，不返回完整粗尺度标签、不修改推理输出、不新增参数。它仍包含可能冗余的嵌套软约束。',
        'I07与I05使用完全相同的剩余区域、非负与总量约束，但所有位置权重固定为1，且不构造权重头；训练和推理均投影，骨干通过投影反传。初始预测与I05一致，不等于B2。I05/I07差异包括学习分配及其新增参数，不宣称等容量。',
        'I08在I03的原四轮内部保留0/1/2/3条detach历史预测，并仅在第4轮最终输出后执行I05投影。池内存储校正前的各轮预测，以可见F回填；投影不反馈到历史池、不改变路由执行数。两模块分别隔离初始化，同时通过最终F任务损失学习，无额外粗尺度损失。',
        'I09是在I08基础上替换为I02自由尺度执行：FMC融合H0；每轮依据未修正状态硬选一个尺度与原生Top2专家；RMS匹配受控差分后叠加因果历史读取，再在被选尺度上注入对应观测并执行专家；恢复F网格后写入detach历史。最终只执行一次I05学习投影，不把投影结果写回历史。无完整粗尺度标签、无额外一致性损失。尺度头解冻增加可训练参数，训练使用所选分支直通梯度；初值相同不等于训练轨迹相同。',
        'I10仅在I09上增加last_round_fine=True：前三轮每样本仍可选F/M/C，无单调或必须覆盖约束；第四轮训练与推理均通过合法尺度mask只允许F，冻结第四尺度头。其余受控差分、历史池和最终投影保持原样；全部参数初值一致，总参数相同，可训练参数因末轮尺度头冻结而减少。I10从公共随机初始化训练100轮，不加载I09已训练权重，不使用损失反馈教师。',
        'I11在I10上仅增加first_three_unique=True。每个样本前三轮根据已选次数屏蔽已用尺度，候选数依次为3/2/1；第四轮单独允许F，不受此前已用F限制。合法路径仅FMCF、FCMF、MFCF、MCFF、CFMF、CMFF六种。按样本逐轮贪心选择，不是并行执行六条路径。第三轮由前两轮决定，第三/第四尺度头冻结，仅前两头接受原ST梯度；总参数与初值不变，可训练参数比I10少一个尺度头。历史、差分、Top2、学习投影与损失都沿用I10。',
        'I11覆盖约束只保证三尺度都被用于轮内计算，不保证融合更好或逐轮误差下降。H0仍已融合F/M/C，不能将轮内覆盖说成首次获得三尺度数据。I11每窗口仍8次专家执行，网格面积代理为2×(1/16+1/4+1+1)=4.625，与固定CMFF相同；不同专家和路由开销仍影响实际成本。I11从公共初值独立训练，不接续I10权重。依赖I10训练、六套评估及正式报告全部完成后，才在GPU1启动预检与训练。',
        'I12按用户指定固定为CMFF，沿用I08的模型、数据、损失、优化器、seed7和公共初始化；不启用自由尺度或无放回路由。H0三尺度融合、逐轮对应原生观测注入、受控差分、因果预测历史池与末轮学习投影全部保留。它是同配置同seed重新运行，不是新增结构消融，也不是新训练种子；不加载I11训练权重。I12在新套件中单独保存，不覆盖I08；只有显式启动后才训练。',
        '本套件所选各组均从头训练，公共初值逐张量与旧B2核对；I03历史模块、I05一致性权重头在独立fork RNG下新增，披露新增容量，不声称与B2等参数。I04不增加可训练参数。无FFN、三平面或额外视图；I04/I05各自只加入下述一致性干预。',
        '本套件所选组物理GPU1串行；各run日志含每15秒GPU状态，未启用周期线程栈采样。旧B2存在历史中断/换卡记录，比较数值协议一致不代表执行历程完全相同。',
        f'六套协议：`{json.dumps(protocol,ensure_ascii=False)}`。训练四族逐epoch重采样，val/test固定且分别偏移基础mask seed 20000/30000；测试mask复测不是独立训练种子。','',
        '### 数据来源与复现','',f'计划 `{suite/"plan.json"}`；源码 `{suite/"source_snapshot"}`；初始化 `{suite/"initialization_audit.json"}`；预检 `{suite/"acceptance/preflight.json"}`。',
        f'恢复：`python -u {suite/"source_snapshot/scripts/v25/run_taxibj_input_communication.py"} --suite {suite}`，外部设置CUDA_VISIBLE_DEVICES=1和OMP/MKL/OPENBLAS线程2；完成组跳过，完整队列结束后tmux退出。','',
        '|组|配置|run|','|---|---|---|']
    lines += [f'|{m}|`{c}`|`{r}`|' for m,c,r in sources]
    if any(int(m[1:]) >= 13 for m in methods if m.startswith('I')):
        lines += ['', '### I13–I22 的独立辅助约束', '',
            'I13–I22保持最终F缺失位置L1与候选均衡不变。独立visible_consistency_aux读取投影前x_hat_raw_consistency；只有无投影I16读取x_hat_main。先回填可见F，按数据sum/mean规则聚合，仅对m_m/m_c可见的x_m_obs/x_c_obs计算L1；sum误差除面积4/16，按每尺度可见数平均后再平均两尺度，空尺度贡献0。禁止完整粗尺度目标、detach或与旧I06重复计损失。',
            '除I14前三轮自由、末轮F外均为CMFF。I13/I14/I15/I16/I19/I20/I21/I22的可见辅助系数0.1，I17为0.03，I18为0.3。I15移除历史参数，I16移除投影头，I19不构造学习分配头，参数差异见结果表。',
            'I20只在历史读取输出后乘0.5，将最大注入从0.1倍隐藏RMS降至0.05倍；池、读写顺序和detach不变。I21采用communication=none，仍逐轮顺序传递隐藏状态并读取历史，差分门保留但冻结。I22在原损失上额外使用相同F监督掩码计算投影前L1，系数0.1。',
            '投影算法及可见F保留、非负剩余总量约束不变。辅助损失反传到骨干。投影前诊断明确使用原始字段；第四轮普通读出已被投影覆盖，不能替代原始字段。无粗尺度约束的F缺失区与受约束区分别记录误差及有效计数；空区域显示—。']

    if set(methods) & {'I23','I24'}:
        lines += ['', '### I23/I24：I14投影前后双重监督', '',
            '两组都从公共初始化独立训练，沿用I14的FMC H0、前三轮自由末轮F、原生共享Top2、受控差分、历史池与学习投影。I23任务损失为0.9×最终F L1＋0.1×投影前F L1；I24为0.7×最终F L1＋0.3×投影前F L1。两项均使用原F缺失监督掩码；保留0.1×投影前可见M/C损失与原候选均衡。',
            'lambda_final_f默认1，旧I01–I22行为不变；新组仅重新分配两项F任务权重，权重和为1。I22则是在完整最终F任务上额外加0.1原始F，且采用固定CMFF；它不是I23的严格单因素对照。I23/I24主要对照I14及彼此。总参数、推理投影和每窗口8次专家执行不变，不载入I14训练权重。',
            '按用户要求在I18训练及六套评估完成后、I19开始前插入。边界控制器仅暂停原队列调度进程，保留原队列GPU1锁；I18评估子进程自然完成后才预检和训练I23/I24。新批结束后恢复原调度器运行I19–I22。前序I13–I18作为只读参考；尚未运行的I19–I22不属于新批报告完成条件。失败时记录原因并停留在边界，不无限重试。']
    if any(f'I{i}' in methods for i in range(30,42)):
        lines += ['', '### I30–I41：通信、尺度与原始预测能力', '',
            'I30去历史，I31去差分，I32固定CMFF；I33同时去历史与差分。I38/I39/I40分别为I32去历史、去差分、两者都去掉。结合旧I25组成尺度自由/固定×历史开/关×差分开/关的2×2×2对照；各组从公共初始化独立训练，旧I25只读引用。',
            '自由尺度无差分组同时设communication=none和delta_steps=[]，防止探索类构造器按delta_steps重新开启差分；差分门保留但冻结。去历史组移除历史读取器参数，固定CMFF冻结原有尺度头，不能称为完全等可训练参数对照。每轮仍只执行原生Top2。',
            'I34固定0.5投影前F+0.5投影后F，I35固定0.8/0.2，两组均移除可学习β标量。I36为I25加0.1投影前可见F重建L1；I37为I34加同一项。可见F损失直接读取回填前x_hat_raw_consistency，对m_f可见位置与x_f_obs比较，独立按可见数归一化，空可见集为零，不读取隐藏真值；不修改推理。此为原始预测重建探索，并非完整复制SAITS的对角注意力或多阶段重建。',
            'I41仅将历史注入gain设为0.5，参数、历史池内容、梯度截断与读写次序不变。所有组保留0.1可见M/C辅助和0.01候选均衡，继续记录投影前后、粗约束覆盖/未覆盖缺失误差、有效计数、路径、历史幅度、损失权重及成本。',
            '初始保留参数逐项核验与冻结I25初始化一致；不是加载已训练I25权重微调。先按原门槛ID≤8.8709筛选，另展示相对I25的ID变化、三类等权OOD及最差类别；不隐去不合格结果。单seed且OOD已多次用于开发筛选，须另用新种子和未参与筛选条件确认。']
    if 'I25' in methods:
        lines += ['', '### I25：可学习的投影前后F损失分配', '',
            '沿用I24模型与协议，从公共初始化独立训练。新增一个全局标量a，β=sigmoid(a)，a初值log(0.3/0.7)，β初值0.3。F任务为(1−β)×投影后缺失位置L1＋β×投影前同位置L1；两权重非负且和为1。保留0.1×投影前可见M/C损失及候选均衡，无额外熵正则或权重下界。',
            '标量和骨干通过同一训练损失联合优化，不detach；标量使用主学习率余弦调度且无weight decay。增加1个可训练参数，不改变推理预测和每窗口8次专家执行。完整检查点保存标量及优化器状态；日志记录β、1−β和前后加权损失。',
            '固定β=0.3的I24按用户要求在完成第2轮后暂停，尚未验证，没有best检查点或六套最终结果，不能作为已完成的等预算对照。I25不续接其两轮权重。I14和I23为已完成参考；不因训练总损失下降就认定泛化改善，因为权重本身可以偏向数值较小的损失。',
            f'I24暂停记录：`{suite/"references/I24/paused.json"}`；实际暂停轮数以该记录为准。']
    if 'I26' in methods:
        lines += ['', '### I26：移除最终投影的完整骨干对照', '',
            '从公共初始化独立训练100轮，保留I25的F/M/C融合H0、逐轮对应观测注入、前三轮自由尺度且允许重复、第四轮固定F、共享八专家原生Top2、受控差分及因果历史池。移除最终投影及其分配头，直接使用第四轮预测。',
            '无投影后前后F预测相同，(1−β)L_F＋βL_F严格合并为单项L_F，β无有效梯度，因此移除该冗余标量。保留0.1×仅可见M/C面积归一化L1，source=prediction；先回填可见F后聚合，不读取完整粗尺度真值。均衡系数仍0.01，F任务掩码不变。参数减少如实披露，不宣称等容量。',
            'I26对I25检验移除投影及其相关监督分解后的整体影响；I26对I14为更直接的有无投影对照，I26对I16仅改变固定CMFF为前三轮自由、末轮F。不能由单seed比较证明投影或路由在所有数据条件下有效。',
            '按用户要求暂停I19，完成I26训练、六套评估和正式报告后从完整last恢复I19并继续I20–I22。原I19暂停记录保存在references/I19/paused.json；本报告不将未完成的I19–I22计作已完成对照。',
            '无投影模型的raw/post分区诊断读取同一预测，二者相等、修正量为零是预期结果；不会再强制满足可见M/C总量或未知区域非负约束，软损失不等价于硬约束。']
    if 'I27' in methods:
        lines += ['', '### I27：I25共享专家池的独立池对照', '',
            'I27仅将I25的expert_sharing由shared改为per_step：每轮独立持有T/S/TD/SD/TA/ST/TL/SL八专家，共四套；每轮仍执行原生Top2，每窗口8次调用。路由器原本就分轮独立，此处不改路由器、不新增跨轮FFN或乘性交互。',
            '保留FMC融合H0、逐轮观测注入、前三轮自由尺度且可重复、第四轮F、受控差分、历史池、I05学习投影、0.1可见M/C损失及0.01均衡。保留I25全局可学习β，初始0.3/0.7，从公共初始化独立训练100轮；不加载I25或I26训练权重。',
            '每套专家逐项复制公共共享池初值，参数存储独立，后续独立优化；其余参数初值相同。初始前向与I25一致，但独立池增加三套专家参数及优化器状态，不能称为等参数对照；实际时长、峰值显存及MAE/RMSE一并报告。',
            '本组检验共享专家复用与分轮专门化的影响，不能单独把差异归因于CoE/MoE名称。I26完成训练、六套评估和报告后执行I27，再释放原控制器恢复I19–I22。I26训练不中断，冻结源码不修改。']
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
    if set(methods)&{'I04','I05','I06','I07','I08','I09','I10','I11','I12'}:
        lines+=['','### 一致性监督与输出约束诊断','','|组|协议|M/C聚合观测残差|投影前M/C残差|约束覆盖比例|修正幅度|权重最小/最大|','|---|---|---|---|---|---|---|']
        for method in methods:
            for label,ev in evals[method].get('sets',{}).items():
                m=ev['metrics'];after='/'.join(fmt(m.get('coe_v25_observed_sum_residual_'+s)) for s in ('M','C'))
                before='/'.join(fmt(m.get('coe_v25_consistency_before_observed_sum_residual_'+s)) for s in ('M','C'))
                weights='/'.join(fmt(m.get('coe_v25_consistency_weight_'+k)) for k in ('min','max'))
                lines.append(f'|{method}|{label}|{after}|{before}|{fmt(m.get("coe_v25_consistency_constrained_fraction"))}|{fmt(m.get("coe_v25_consistency_correction_abs"))}|{weights}|')
        lines+=['','只有聚合残差下降不能证明细粒度MAE改善。I05的校正位于最终读出之后，不能将校正后输出与前几轮读出当作未经处理的逐轮修复轨迹。全量训练损失分量与约束组数见curves.json、evaluations.json；缺失诊断显示为“—”，不补造。']
    if any(m in evals for m in [f'I{i:02}' for i in range(13,28)]):
        lines += ['', '### 原始预测与分区域诊断', '',
                  '|组|协议|区域|有效F缺失数|原始MAE/RMSE|最终MAE/RMSE|', '|---|---|---|---|---|---|']
        diagnostics = []
        for method, evaluation in evals.items():
            for label, ev in evaluation.get('sets', {}).items():
                metrics = ev['metrics']
                if 'combination_raw_all_count' not in metrics: continue
                for area in ('all','constrained','unconstrained'):
                    count = metrics['combination_raw_'+area+'_count']
                    values = ['/'.join(fmt(metrics.get(f'combination_{stage}_{area}_{k}')) for k in ('mae','rmse'))
                              if count else '—' for stage in ('raw','post')]
                    lines.append(f'|{method}|{label}|{area}|{int(count)}|{values[0]}|{values[1]}|')
                diagnostics.append({'method':method, 'protocol':label,
                    **{k:v for k,v in metrics.items() if k.startswith(('combination_', 'l_visible_consistency', 'l_raw_f', 'l_visible_raw_f', 'visible_raw_f_count'))}})
        write(assets/'combination_diagnostics.json', diagnostics)
        lines += ['', '|组|协议|可见M原始/最终残差|可见C原始/最终残差|辅助损失/加权|raw F损失/加权|', '|---|---|---|---|---|---|']
        for d in diagnostics:
            residuals = ['/'.join(fmt(d.get(f'combination_{stage}_visible_{scale}_mae')) for stage in ('raw','post')) for scale in ('m','c')]
            aux = '/'.join(fmt(d.get(k)) for k in ('l_visible_consistency_aux','l_visible_consistency_aux_weighted'))
            raw = '/'.join(fmt(d.get(k)) for k in ('l_raw_f','l_raw_f_weighted'))
            lines.append(f'|{d["method"]}|{d["protocol"]}|{residuals[0]}|{residuals[1]}|{aux}|{raw}|')
    if any(f'I{i}' in methods for i in range(30,42)):
        lines += ['', '### 新组原始预测监督与可见F重建轨迹', '',
            '|组|epoch|投影前F损失|投影后F损失|β（固定/可学习）|可见F重建|加权可见F重建|', '|---|---|---|---|---|---|---|']
        for method in (f'I{i}' for i in range(30,42)):
            for row in histories.get(method,[]):
                if row['epoch'] != 1 and row['epoch'] % 5: continue
                v=row['train']
                beta=v.get('f_loss_raw_weight', .8 if method=='I35' else .5 if method in ('I34','I37') else None)
                lines.append(f'|{method}|{row["epoch"]}|{fmt(v.get("l_raw_f"))}|{fmt(v.get("l_final"))}|{fmt(beta)}|{fmt(v.get("l_visible_raw_f"))}|{fmt(v.get("l_visible_raw_f_weighted"))}|')
    if 'I25' in methods:
        lines += ['', '### 学习权重轨迹', '', '|epoch|训练平均投影前权重β|训练平均投影后权重|验证时β|', '|---|---|---|---|']
        for row in histories.get('I25',[]):
            if row['epoch']==1 or row['epoch']%5==0:
                lines.append(f'|{row["epoch"]}|{fmt(row["train"].get("f_loss_raw_weight"))}|{fmt(row["train"].get("f_loss_final_weight"))}|{fmt((row.get("val") or {}).get("f_loss_raw_weight"))}|')
    if 'I27' in methods:
        lines += ['', '### I27学习权重轨迹', '', '|epoch|训练平均β|训练平均1−β|验证β|', '|---|---|---|---|']
        for row in histories.get('I27',[]):
            if row['epoch']==1 or row['epoch']%5==0:
                lines.append(f'|{row["epoch"]}|{fmt(row["train"].get("f_loss_raw_weight"))}|{fmt(row["train"].get("f_loss_final_weight"))}|{fmt((row.get("val") or {}).get("f_loss_raw_weight"))}|')
    lines+=['','### 完成与失败审计' ,'',('；'.join(issues) if issues else '已有凭据/配置/检查点/主指标未发现核验异常；未完成组以上表为准。')]
    for r in records:
        if not r['complete']:lines.append(f'- {r["method"]}：未完成。')
        if r['failure']:lines.append(f'- {r["method"]}历史失败：`{r["failure"]}`。')
    lines+=['','## 3. 结果分析与建议','','### 预设配对的实测变化','','负数表示误差改善；不省略负结果。','','|配对|协议|MAE变化%|RMSE变化%|','|---|---|---|---|']
    pairs=[(m,ref) for m in methods for ref in ('B2','B1')]
    if 'I01' in methods:pairs.append(('I01','A2'))
    if 'I04' in methods and 'I05' in methods:pairs.append(('I05','I04'))
    pairs += [('I06','I04'),('I05','I06'),('I07','I06'),('I05','I07'),('I08','I03'),('I08','I05'),('I09','I08'),('I09','I02'),('I09','I03'),('I09','I05'),('I09','I07'),('I10','I09'),('I10','I08'),('I10','I02'),('I10','I03'),('I10','I05')]
    pairs += [('I11','I10'),('I11','I08'),('I11','I09'),('I11','I03'),('I11','I05')]
    pairs += [('I12','I08'),('I12','I11'),('I12','I10'),('I12','I03'),('I12','I05')]
    pairs += [('I13','I08'),('I14','I10'),('I14','I13'),('I15','I05'),('I15','I13'),('I16','I03'),('I16','I13')] + [(f'I{i:02}','I13') for i in range(17,23)]
    pairs += [('I23','I14'),('I24','I14'),('I24','I23'),('I23','I06'),('I24','I06'),('I23','I16'),('I24','I16')]
    pairs += [('I25','I14'),('I25','I23'),('I25','I06'),('I25','I16')]
    pairs += [('I26','I25'),('I26','I14'),('I26','I16'),('I26','I06')]
    pairs += [('I27','I25')]
    pairs += [(f'I{i}','I25') for i in range(30,42)]
    pairs += [('I33','I30'),('I33','I31'),('I37','I34'),('I37','I36'),
              ('I38','I32'),('I39','I32'),('I40','I32'),
              ('I38','I30'),('I39','I31'),('I40','I33')]
    for a,b in pairs:
        ea,eb=evals.get(a,{}).get('sets',{}),evals.get(b,{}).get('sets',{})
        for lab in protocol['evaluations']:
            if lab in ea and lab in eb:
                x,y=ea[lab]['metrics'],eb[lab]['metrics'];lines.append(f'|{a}/{b}|{lab}|{100*(x["mae"]/y["mae"]-1):+.3f}|{100*(x["rmse"]/y["rmse"]-1):+.3f}|')
    if all(m in evals for m in ('B2','I03','I05','I08')):
        lines+=['','### 历史与投影的交互诊断','','交互量=I08−I03−I05＋B2，在相同协议的原始误差单位上计算；负值仅表示该指标上的联合收益超过加和预期，不构成单种子显著性结论。','','|协议|MAE交互量|RMSE交互量|','|---|---|---|']
        for label in protocol['evaluations']:
            sets=[evals[m].get('sets',{}).get(label) for m in ('I08','I03','I05','B2')]
            if all(sets):
                values=[sum(sign*row['metrics'][key] for sign,row in zip((1,-1,-1,1),sets)) for key in ('mae','rmse')]
                lines.append(f'|{label}|{fmt(values[0])}|{fmt(values[1])}|')
    if any(m in evals for m in [f'I{i:02}' for i in range(13,28)]):
        ranking = combination_ranking(evals)
        write(assets/'combination_ranking.json', ranking)
        lines += ['', '### 预设 ID 门槛与三类 OOD 等权排序', '',
            'ID测试MAE≤8.8709才进入候选。二元与形态各先平均两套相对I08误差比，再与三元类别等权平均；不是把五套协议直接平均。排序依次为OOD MAE比、OOD RMSE比、ID MAE。最差类别变化为三类中最大变化，负数表示所有类别均改善。I12是I08同seed重复，不能作为独立证据。', '',
            '|组|ID MAE|门槛|三类MAE比（二元/形态/三元）|OOD MAE比|OOD RMSE比|最差类别MAE/RMSE变化%|', '|---|---|---|---|---|---|---|']
        for r in ranking:
            ratios = '/'.join(fmt(v) for v in r['mae_categories'].values())
            lines.append(f'|{r["method"]}|{fmt(r["id_mae"])}|{"通过" if r["eligible"] else "未达"}|{ratios}|{fmt(r["mae_score"])}|{fmt(r["rmse_score"])}|{r["mae_worst_change_pct"]:+.3f}/{r["rmse_worst_change_pct"]:+.3f}|')
        candidates = [r for r in ranking if r['eligible'] and r['method'] != 'I12']
        if candidates and final:
            best = candidates[0]
            lines += ['', f'按预设规则，当前合格模型中排序第一为{best["method"]}，OOD MAE比{best["mae_score"]:.6f}、ID MAE {best["id_mae"]:.6f}。']
            if best['method'] not in {j['method'] for j in js.values()}:
                lines += ['本批新增组未超过既有合格参考的该项折中，保留既有参考，不强行推荐新组。']
        elif final:
            lines += ['', '没有模型通过预设ID门槛，不推荐本批新模型。']
        else:
            lines += ['', '本批仍未完成，以上为已有完整六套评估的阶段排序，不代表最终推荐。']
        lines += ['', '本批将OOD用于探索筛选；最终确认仍需新种子与未用于筛选的条件。I21/I13检验完整组合中差分的增量，不单独证明所有骨干上的历史与差分都冗余。辅助约束的作用由配对数据判断，不能只看投影后残差为零。']
    if final:
        for name in methods:
            changes={lab:100*(ev['metrics']['mae']/evals['B2']['sets'][lab]['metrics']['mae']-1) for lab,ev in evals[name]['sets'].items()};ood=[v for k,v in changes.items() if k!='in_distribution'];wins=sum(v<0 for v in ood)
            lines+=['',f'{name}相对B2：ID MAE变化{changes["in_distribution"]:+.2f}%；五套OOD中{wins}套MAE改善，变化范围{min(ood):+.2f}%至{max(ood):+.2f}%。']
    else:lines+=['','本批尚未完成；已出现的局部结果不代表本套件最终排序。']
    lines+=['','### 解释边界与下一步','',
        'I01/B2主要隔离初始入口，但同时改变初始状态、早期差分来源及初始数值后备；若改善，只支持改变入口有帮助，不足以证明原泛化退步由差分独立造成。仍需同入口无差分对照检验交互。I01/A2则包含逐轮可见M/C的额外信息，不能归为纯算法收益。',
        'I02/B2比较的是当前硬尺度估计器与完整差分的组合，若路径集中，需要固定该路径对照；没有对应同协议自由尺度无差分结果时，不能称为二者互补。不得以BikeNYC的S01结果替代TaxiBJ配对。',
        'I03/B2包含新增历史模块容量与绝对预测读取，不能单凭改善证明历史与差分不冗余；需要历史无差分、仅最近历史、等容量分支等后续控制。历史权重不是因果重要性，零初始化不保证训练后不破坏泛化。',
        'I04/B2检验完整粗尺度监督的增量；I05/B2检验仅可见观测约束下的学习校正。I05/I04同时改变监督范围、输出约束、计算及参数量，不能单独证明自适应分配优于同信息量软约束。I06/I07补充可见观测软损失和固定投影，仍缺等容量修正头。',
        'I06/I04控制标签可见范围但改变有效监督位置的归一化；I07/I06对比同可见信息、无新增参数下的软损失和固定投影，但损失权重与推理约束也不同。I05/I07检查学习分配的增量，仍有新增参数因素。I08/I03及I08/I05检验模块组合；旧I01–I12没有历史无差分对照；新I21/I13补充完整组合中的差分消融，仍不是普遍独立性证明。',
        'I05只处理本批精确、对齐、相容的嵌套非负观测。固定欧氏投影的平方误差不增性质不自动适用于学习权重，也不保证MAE不增。约束组分解属于实现机制，独立新颖性和稳定泛化须由后续控制支持。',
        'I09/I08主要检查自由尺度训练在历史加投影基础上的增量；与I02相比同时加入历史和投影，不能独立归因为其中一个模块。本批缺少自由尺度×历史×投影的完整因子设计；若测试路径仍集中CMFF，不能把收益归因于按样本动态切换尺度。单种子组合成败不是普遍结论。',
        'I10/I09对比须注意：I09由用户在64轮后暂停，best来自epoch25；I10计划从头100轮。I09六套测试完成仅代表暂停检查点已评估，不代表完成原100轮训练。末轮固定F若改善，支持该结构限制在本次训练中有帮助，不能据此证明尺度切换是此前反弹的唯一原因；同时改变训练轨迹且预算不等。I10最终报告完成只指本套件新增训练结束，不能把暂停参考计为已完成100轮。',
        'I11/I10隔离前三轮不重复覆盖约束，但也改变计算尺度分布、路由选择自由度和可训练尺度头数；I11/I08的尺度计数及网格面积代理一致，主要比较学习顺序与固定CMFF。若I11最终仍集中CMFF，不能据此宣称按样本动态顺序有收益，应进一步匹配固定路径或路由训练对照。只有六条路径合法不代表六条都被实际使用。',
        '单seed7、同率训练测试，不支持显著性或稳定跨数据集结论；五套OOD及mask复测不是五个独立训练样本。只按ID验证选checkpoint，研究筛选仍已参考OOD，正式确认应保留新的条件/种子。中间轮无监督，逐轮读出不保证单调。MAPE受零值影响不作为主判据。',
        '所有成本是实际运行记录，未严格GPU独占；累计训练分钟不含故障等待与评估，不能等同墙钟时长。先逐协议检查已配置各组与B2和B1的改善/退步，再决定补机制对照，不根据单个最好结果直接更换主线。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n');write(suite/'report_path.json',{'path':str(path),'complete':final,'issues':issues});return path
