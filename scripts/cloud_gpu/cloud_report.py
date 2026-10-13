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
    rows=[];curves={};evals={};issues=[];results=[];families=[]
    for name,j in jobs.items():
        row,h,e,bad=collect(suite,name,j);rows.append(row);curves[j['method']]=h;evals[j['method']]=e;issues+=bad
        for label,ev in e.get('sets',{}).items():
            m=ev['metrics'];results.append({'method':j['method'],'protocol':label,'mae':m['mae'],'rmse':m['rmse'],'samples':ev['samples'],'mask_seed':ev['effective_mask_seed'],'ms_per_sample':m.get('forward_ms_per_sample_per_rank'),'expert_calls':m.get('coe_expert_execution_count'),'area':m.get('coe_expert_grid_equivalents')})
            for key,val in m.items():
                if key.startswith('coe_family_') and key.endswith('_mae'):
                    families.append({'method':j['method'],'protocol':label,'family':key[len('coe_family_'):-4],'mae':val,'rmse':m.get(key[:-3]+'rmse'),'count':m.get(key[:-3]+'count')})
    complete=bool(rows) and all(r['complete'] for r in rows) and not issues
    first=next(iter(jobs.values()));cfg=first['config'];protocol=cfg['experiment_plan']['protocol']
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d');phase='完整' if complete else '阶段'
    stem=f'{date}_v25_cloud_{first["dataset"]}_{suite.name}';assets=dest/(stem+'_assets');assets.mkdir(exist_ok=True)
    for name,value in [('summary.json',rows),('curves.json',curves),('evaluations.json',evals),('audit.json',{'complete':complete,'issues':issues})]:write(assets/name,value)
    for name,data in [('protocols.csv',results),('families.csv',families)]:
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
           '|组|可见尺度 / H0|轮次尺度|各轮Top-k|差分输入轮|历史池|','|---|---|---|---|---|---|']
    for j in jobs.values():
        c=j['config'];coe=c['model']['coe'];s=coe.get('scale_delta_exploration',{});delta=s.get('delta_steps',[2,3,4] if coe['core_validation']['communication']!='none' else [])
        lines.append(f'|{j["method"]}|{c["data"]["input_scales"]} / {"F" if c["data"]["input_scales"]=="F" else coe["initial_observation_scales"]}|{s.get("path","CMFF")}|{s.get("top_k_per_step",[2]*4)}|{delta}|{coe.get("observed_history",{}).get("mode","无")}|')
    lines += ['', 'A1/A2只提供F观测，B1/B2提供真实预构造F/M/C观测；A2/B2使用受控差分。I01相对B2只将H0改为F；I02只改自由尺度选择；I03只加自适应预测历史池（无乘积交互）。S01–S03相对B1改变尺度/Top-k；S04完整差分复验；S05–S07仅在第2/3/4轮启用差分；S08无差分CFFF；S09在CFFF第4轮启用同尺度差分。',
      '八种异构专家四轮共享；原生执行选中Top-k，固定尺度调度与动态专家选择分开。各轮读取对应尺度原始观测，隐藏状态恢复到F网格。差分经过RMS匹配、±0.1条件门，仅修正专家输入。自由尺度每轮硬选一种，不含软预热。',
      'F/M/C在数据预处理阶段构造，各尺度独立遮挡；未观测槽位在前向中被掩码清除。粗尺度可见聚合观测可能包含F上未观测位置的信号，这是多尺度任务提供的额外信息；B与A比较不能仅归因为架构能力。TaxiBJ/BikeNYC求和，CHAP求均值。BikeNYC空间补齐行仍按原协议评分，CHAP原始缺测限制见数据manifest。',
      f'缺失率 {protocol["rate"]}，四类混合训练mask {protocol["train_families"]}；不是纯random点缺失。{cfg["train"]["epochs"]} epoch，batch{cfg["data"]["batch_size"]}，验证间隔{cfg["train"]["val_epoch"]}；lr={cfg["train"]["lr_main"]}，scheduler={cfg["train"]["scheduler"]}。仅ID验证MAE选择best。',
      '同批共享参数初始化逐张量哈希核验；各组从头训练。恢复包含优化器、调度器、AMP与随机数状态，从last完整epoch继续。跨硬件数值一致性未获保证。',
      f'冻结计划 `{suite/"plan.json"}`；逐组配置 `{suite/"configs"}`；源码 `{suite/"source_snapshot"}`；初始化审计 `{suite/"initialization_audit.json"}`；运行设备 `{suite/"runtime_launch.json"}`。',
      '', '## 2. 完整结果、训练曲线与诊断','','|组|状态|epoch|best epoch|验证MAE|训练分钟|总/可训参数|峰值GiB|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["state"]}|{r["epochs"]}|{r["best_epoch"]}|{fmt(r["val_mae"])}|{fmt(r["minutes"])}|{r["params"]}/{r["trainable"]}|{fmt(r["peak_gib"])}|' for r in rows]
    lines += ['','|组|协议|MAE|RMSE|ms/窗口|专家调用|面积代理|mask seed|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{fmt(r["ms_per_sample"])}|{fmt(r["expert_calls"])}|{fmt(r["area"])}|{r["mask_seed"]}|' for r in results]
    lines += ['','|组|协议|缺失族|MAE|RMSE|计数|','|---|---|---|---|---|---|']
    lines += [f'|{r["method"]}|{r["protocol"]}|{r["family"]}|{fmt(r["mae"])}|{fmt(r["rmse"])}|{r["count"]}|' for r in families]
    lines += ['',f'![训练与验证曲线]({assets.name}/curves.png)','',f'完整路由/尺度路径、门值、修正量、历史池诊断及分族指标：[evaluations.json]({assets.name}/evaluations.json)。逐epoch：[curves.json]({assets.name}/curves.json)。','', '### 失败与缺失','']
    lines += ['- '+x for x in issues] or ['未发现已完成结果的配置、检查点、指标、epoch序列审计异常。']
    for r in rows:
        if not r['complete']:lines.append(f'- {r["method"]}：{r["state"]}，{r["epochs"]}epoch。')
        if r['failure']:lines.append(f'- {r["method"]} 历史失败：`{r["failure"]}`。')
    lines += ['','## 3. 分析、证据边界与下一步','','误差变化=(新/参考−1)×100%，负数更好；所有协议保留，包含负结果。','', '|对照|协议|MAE变化%|RMSE变化%|','|---|---|---|---|']
    pairs=[('A2','A1'),('B1','A1'),('B2','B1'),('B2','A2'),('I01','B2'),('I02','B2'),('I03','B2')]+[(f'S{i:02}','B1') for i in range(1,8)]+[('S04','B2'),('S09','S08')]
    for a,b in pairs:
        ea=evals.get(a,{}).get('sets',{});eb=evals.get(b,{}).get('sets',{})
        for label in protocol['evaluations']:
            if label in ea and label in eb:
                vals=[100*(ea[label]['metrics'][key]/eb[label]['metrics'][key]-1) if eb[label]['metrics'][key] else None for key in ('mae','rmse')]
                lines.append(f'|{a}/{b}|{label}|{fmt(vals[0])}|{fmt(vals[1])}|')
    lines += ['', '单训练种子不能证明稳定收益；测试mask复测不是训练种子复验。ID与未见组合/几何/三重缺失分别判断，不用平均名次掩盖退步。增大Top-k同时增加计算，不是等成本对照。门值、尺度路径和单位置消融只提供机制线索，不能证明每轮误差下降。',
      '粗尺度输入是否有助于泛化、差分是否与跨尺度状态变化冲突，需要匹配训练预算的B1/B2及位置控制；I03同时保留差分时，不能证明历史池与差分不存在冗余。正式结论仍需seed17/27与跨数据集验证。',
      '本报告不自动把旧服务器结果混入新批次。硬件/环境、训练预算或mask协议不同需披露；运行耗时不含全部失败和等待时间。']
    path=dest/(stem+'_'+phase+'分析报告.md');path.write_text('\n'.join(lines)+'\n')
    write(suite/'report_path.json',{'path':str(path),'complete':complete,'issues':issues});return path
