"""Evidence-audited batch report, methods first; no original-server artifacts needed."""
import csv,json,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write
from report_scale_delta_exploration import collect,fmt


def export_report(suite,jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suite=Path(suite);plan=load(suite/'plan.json');dest=Path(plan['project_root'])/'experments_report';dest.mkdir(parents=True,exist_ok=True)
    rows=[];curves={};evals={};issues=[];results=[];families=[];consistency=[]
    for name,j in jobs.items():
        row,h,e,bad=collect(suite,name,j);rows.append(row);curves[j['method']]=h;evals[j['method']]=e;issues+=bad
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];results.append({'method':j['method'],'protocol':label,'mae':m['mae'],'rmse':m['rmse'],'samples':ev['samples'],'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank'),'expert_calls':m.get('coe_expert_execution_count'),'area':m.get('coe_expert_grid_equivalents')})
            if j['config']['model']['coe'].get('observation_consistency',{}).get('enabled',False):
                pooling=j['config']['data']['scales']['pooling_mode'];prefix='coe_v25_consistency_'
                consistency.append({'method':j['method'],'protocol':label,
                    'before_m':m.get(prefix+'before_observed_'+pooling+'_residual_M'),
                    'before_c':m.get(prefix+'before_observed_'+pooling+'_residual_C'),
                    'after_m':m.get(prefix+'after_observed_'+pooling+'_residual_M',m.get('coe_v25_observed_'+pooling+'_residual_M')),
                    'after_c':m.get(prefix+'after_observed_'+pooling+'_residual_C',m.get('coe_v25_observed_'+pooling+'_residual_C')),
                    'constrained_fraction':m.get(prefix+'constrained_fraction'),
                    'correction_abs':m.get(prefix+'correction_abs'),
                    'weight_min':m.get(prefix+'weight_min'),'weight_max':m.get(prefix+'weight_max')})
            for key,val in m.items():
                if key.startswith('coe_family_') and key.endswith('_mae'):
                    families.append({'method':j['method'],'protocol':label,'family':key[len('coe_family_'):-4],'mae':val,'rmse':m.get(key[:-3]+'rmse'),'count':m.get(key[:-3]+'count')})
    complete=bool(rows) and all(r['complete'] for r in rows) and not issues
    first=next(iter(jobs.values()));cfg=first['config'];protocol=cfg['experiment_plan']['protocol']
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d');phase='完整' if complete else '阶段'
    stem=f'{date}_v25_cloud_{first["dataset"]}_{suite.name}';assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for name,value in [('summary.json',rows),('curves.json',curves),('evaluations.json',evals),('audit.json',{'complete':complete,'issues':issues})]:write(assets/name,value)
    for name,data in [('protocols.csv',results),('families.csv',families),('consistency.csv',consistency)]:
        if data:
            with (assets/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method,h in curves.items():
        if not h:continue
        axes[0].plot([x['epoch'] for x in h],[x['train']['mae'] for x in h],label=method)
        v=[x for x in h if x.get('val')];axes[1].plot([x['epoch'] for x in v],[x['val']['mae'] for x in v],label=method)
    for ax,title in zip(axes,['Training MAE','ID validation MAE']):
        ax.set(title=title,xlabel='Epoch',ylabel='MAE');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(assets/'curves.png',dpi=150);plt.close(fig)
    lines=[f'# {stem} {phase}分析报告','','## 1. 具体做法、对照与复现配置','',f'本批 {len(jobs)} 组；数据集 {first["dataset"]}；训练 seed{cfg["seed"]}。完成状态见逐组表。','',
           '|组|可见尺度 / H0|轮次尺度|各轮Top-k|差分输入轮|历史池|一致性机制|','|---|---|---|---|---|---|---|']
    for j in jobs.values():
        c=j['config'];coe=c['model']['coe'];s=coe.get('scale_delta_exploration',{});delta=s.get('delta_steps',[2,3,4] if coe['core_validation']['communication']!='none' else [])
        cs=coe.get('observation_consistency',{});mode=cs.get('mode','无') if cs.get('enabled',False) else '无'
        if mode in ('full_target_loss','observed_loss'):mode+='; λ='+str(c['loss'].get('lambda_observation_consistency',.1))
        if mode=='residual_projection':mode+='; weights='+cs.get('weight_mode','learned')+'; weight_bound='+str(cs.get('weight_bound',2.))
        lines.append(f'|{j["method"]}|{c["data"]["input_scales"]} / {"F" if c["data"]["input_scales"]=="F" else coe["initial_observation_scales"]}|{s.get("path","CMFF")}|{s.get("top_k_per_step",[2]*4)}|{delta}|{coe.get("observed_history",{}).get("mode","无")}|{mode}|')
    lines += ['', 'A1/A2只提供F观测，B1/B2提供真实预构造F/M/C观测；A2/B2使用受控差分。I01相对B2只将H0改为F；I02只改自由尺度选择；I03只加自适应预测历史池（无乘积交互）。S01–S03相对B1改变尺度/Top-k；S04完整差分复验；S05–S07仅在第2/3/4轮启用差分；S08无差分CFFF；S09在CFFF第4轮启用同尺度差分。',
      '八种异构专家四轮共享；原生执行选中Top-k，固定尺度调度与动态专家选择分开。各轮读取对应尺度原始观测，隐藏状态恢复到F网格。差分经过RMS匹配、±0.1条件门，仅修正专家输入。自由尺度每轮硬选一种，不含软预热。',
      'F/M/C在数据预处理阶段构造，各尺度独立遮挡；未观测槽位在前向中被掩码清除。粗尺度可见聚合观测可能包含F上未观测位置的信号，这是多尺度任务提供的额外信息；B与A比较不能仅归因为架构能力。TaxiBJ/BikeNYC求和，CHAP求均值。BikeNYC空间补齐行仍按原协议评分，CHAP原始缺测限制见数据manifest。',
      f'缺失率 {protocol["rate"]}，四类混合训练mask {protocol["train_families"]}；不是纯random点缺失。{cfg["train"]["epochs"]} epoch，batch{cfg["data"]["batch_size"]}，验证间隔{cfg["train"]["val_epoch"]}；lr={cfg["train"]["lr_main"]}，scheduler={cfg["train"]["scheduler"]}。仅ID验证MAE选择best。',
      '同批共享参数初始化逐张量哈希核验；各组从头训练。恢复包含优化器、调度器、AMP与随机数状态，从last完整epoch继续。跨硬件数值一致性未获保证。',
      f'冻结计划 `{suite/"plan.json"}`；逐组配置 `{suite/"configs"}`；源码 `{suite/"source_snapshot"}`；初始化审计 `{suite/"initialization_audit.json"}`；运行设备 `{suite/"runtime_launch.json"}`。',
    ]
    if any(j['method'] in ('I04','I05','I06','I07','I08') for j in jobs.values()):
        lines += ['', '### I04–I08 做法与监督边界','',
          'I04–I08均以B2的三尺度融合H0、CMFF共享原生Top-2和完整受控差分为基础；仅I08增加I03历史池，不改自由尺度路由。',
          'I04：最终F预测先回填可见F，再按原数据的2×2、4×4空间聚合构造预测M/C。分别对预处理文件中的完整x_m_gt/x_c_gt计算所有位置L1；求和数据分别除面积4、16，均值数据不再除面积。两个尺度损失取平均，按配置λ（默认0.1）加入原F缺失位置任务损失与路由均衡项。完整M/C只作为损失标签，前向始终只读取可见输入；I04不增加模型参数，不保证输出严格满足观测等式。',
          'I05：仅依据可见观测建立剩余区域。可见M总量减内部可见F；可见C总量先减已观测M整块，再减其余区域内的可见F，避免重复扣除。互不重叠的剩余未知区域分别进行带非负约束的加权最小改动投影。可见F保持原值，没有粗尺度约束的位置保留原预测；均值数据按区域面积转成总量后处理。',
          'I05权重头读取最终隐藏状态、以观测RMS归一化的F预测、F/M/C覆盖mask，产生exp(weight_bound×tanh(logit))；默认weight_bound=2。末层零初始化使初始权重为1，即固定欧氏投影，初始输出并不等于B2。新增x_consistency参数独立初始化，公共骨干保持原初值，通过原F任务损失端到端学习；不读取完整F/M/C标签作为前向校正信息，不增加粗尺度目标损失。',
          'I04使用完整粗尺度目标进行额外监督，I05只根据可见观测校正输出。因此两者不是同信息、等监督的纯软约束与硬约束对照。完整M/C标签虽由已有F真值聚合得到，监督方式与损失权重仍改变。I05目前要求精确、对齐、非负且聚合相容的观测，不能外推到任意带噪或非对齐观测。',
          'I06：仅在可见M/C位置，对观测值计算聚合L1；各尺度按可见单元数平均、求和数据除面积，再固定取两尺度均值，空尺度贡献0；λ=0.1，不返回粗尺度完整标签，不增加参数。I07：I05所有投影权重固定为1，不创建权重头，训练仍通过投影向骨干反传。I08：I03历史读取叠加I05最终投影，池中仍存可见F回填并detach的原始逐轮预测，不将最终校正放回历史。',
          '已冻结的I01–I03批次保持原计划和源码；I04/I05在新的独立冻结批次中运行。不能因新增代码就把旧三组完成回执当作新增队列的完成凭据。']
    lines += ['', '## 2. 完整结果、训练曲线与诊断','','|组|状态|epoch|best epoch|验证MAE|训练分钟|总/可训参数|峰值GiB|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["state"]}|{r["epochs"]}|{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in rows]
    lines += ['','|组|协议|MAE|RMSE|ms/窗口|专家调用|面积代理|mask seed|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{fmt(r["expert_calls"])}|{fmt(r["area"])}|{r["mask_seed"]}|' for r in results]
    lines += ['','|组|协议|缺失族|MAE|RMSE|计数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    if consistency:
        lines += ['', '### 最终输出一致性诊断','',
          'M/C残差按物理求和或均值单位报告，仅评价可见粗尺度观测。I04/I06无投影前后之分，最终残差列记录其模型输出；缺少指标用空值标记，不能视作0。I05/I07/I08受约束比例以F缺失位置为分母。',
          '', '|组|协议|M投影前残差|C投影前残差|M最终残差|C最终残差|受约束比例|校正绝对量|权重min/max|',
          '|---|---|---|---|---|---|---|---|---|']
        lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["before_m"])}|{fmt(r["before_c"])}|{fmt(r["after_m"])}|{fmt(r["after_c"])}|{fmt(r["constrained_fraction"])}|{fmt(r["correction_abs"])}|{fmt(r["weight_min"])}/{fmt(r["weight_max"])}|' for r in consistency]
        lines += ['',f'逐epoch完整辅助损失与训练/验证诊断保留于[curves.json]({assets.name}/curves.json)；I04/I06的l_observation_consistency是除面积后的M/C平均L1，与这里的可见物理残差口径不同。']
    lines += ['',f'![训练与验证曲线]({assets.name}/curves.png)','',f'完整路由/尺度路径、门值、修正量、历史池诊断及分族指标：[evaluations.json]({assets.name}/evaluations.json)。逐epoch：[curves.json]({assets.name}/curves.json)。','', '### 失败与缺失','']
    lines += ['- '+x for x in issues] or ['未发现已完成结果的配置、检查点、指标、epoch序列审计异常。']
    for r in rows:
        if not r['complete']:lines.append(f'- {r["method"]}：{r["state"]}，{r["epochs"]}epoch。')
        if r['failure']:lines.append(f'- {r["method"]} 历史失败：`{r["failure"]}`。')
    lines += ['','## 3. 分析、证据边界与下一步','','误差变化=(新/参考−1)×100%，负数更好；所有协议保留，包含负结果。','', '|对照|协议|MAE变化%|RMSE变化%|','|---|---|---|---|']
    pairs=[('A2','A1'),('B1','A1'),('B2','B1'),('B2','A2'),('I01','B2'),('I02','B2'),('I03','B2'),('I04','B2'),('I05','B2'),('I05','I04')]+[(f'S{i:02}','B1') for i in range(1,8)]+[('S04','B2'),('S09','S08')]
    pairs += [('I06','B2'),('I07','B2'),('I08','B2'),('I06','I04'),('I05','I06'),('I07','I06'),('I05','I07'),('I08','I03'),('I08','I05')]
    for a,b in pairs:
        ea=evals.get(a,{}).get('sets',{});eb=evals.get(b,{}).get('sets',{})
        for label in protocol['evaluations']:
            if label in ea and label in eb:
                vals=[100*(ea[label]['metrics'][key]/eb[label]['metrics'][key]-1) if eb[label]['metrics'][key] else None for key in ('mae','rmse')]
                lines.append(f'|{a}/{b}|{label}|{fmt(vals[0])}|{fmt(vals[1])}|')
    if any(j['method'] in ('I04','I05','I06','I07','I08') for j in jobs.values()):
        lines += ['', 'I04/B2检验完整粗尺度目标辅助监督，I05/B2检验观测驱动自适应投影，I05/I04比较两种整体方案；最后一项不能单独归因为软硬约束形式。只计算本批中双方都存在且评估完成的协议。',
          '投影使观测聚合残差接近0不是补全精度或泛化收益的证据；必须同时比较F缺失位置MAE/RMSE、未见缺失协议与成本。可学习权重不自动继承固定欧氏投影相对真值的平方误差不增性质。只有一个训练种子时不宣称稳定改进。',
          'I06/I04检查标签可见范围；I07/I06检查同信息、无新增参数的软损失与固定投影；I05/I07检查学习分配，仍有新增容量因素；I08/I03和I08/I05检查组合增量。本批没有历史无差分对照，也没有等容量普通修正头。']
        if 'B2' not in evals:
            lines += ['本批未包含B2，因此未生成I04/B2或I05/B2数值结论；不能自动借用不同冻结批次或旧服务器结果补齐。']
    if all(m in evals for m in ('B2','I03','I05','I08')):
        lines += ['', '历史×投影交互量=I08−I03−I05＋B2；负值为原始误差单位上的联合增益线索，不是统计显著性。', '', '|协议|MAE交互量|RMSE交互量|','|---|---|---|']
        for label in protocol['evaluations']:
            sets=[evals[m].get('sets',{}).get(label) for m in ('I08','I03','I05','B2')]
            if all(sets):
                values=[sum(sign*row['metrics'][key] for sign,row in zip((1,-1,-1,1),sets)) for key in ('mae','rmse')]
                lines.append(f'|{label}|{fmt(values[0])}|{fmt(values[1])}|')
    lines += ['', '单训练种子不能证明稳定收益；测试mask复测不是训练种子复验。ID与未见组合/几何/三重缺失分别判断，不用平均名次掩盖退步。增大Top-k同时增加计算，不是等成本对照。门值、尺度路径和单位置消融只提供机制线索，不能证明每轮误差下降。',
      '粗尺度输入是否有助于泛化、差分是否与跨尺度状态变化冲突，需要匹配训练预算的B1/B2及位置控制；I03同时保留差分时，不能证明历史池与差分不存在冗余。正式结论仍需seed17/27与跨数据集验证。',
      '本报告不自动把旧服务器结果混入新批次。硬件/环境、训练预算或mask协议不同需披露；运行耗时不含全部失败和等待时间。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'report_path.json',{'path':str(path),'complete':complete,'issues':issues});return path
