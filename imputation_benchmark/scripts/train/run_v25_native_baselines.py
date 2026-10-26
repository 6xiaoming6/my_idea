#!/usr/bin/env python3
"""Frozen F-only native baseline queue, with batch backoff and six OOD protocols."""
from __future__ import annotations
import argparse, copy, datetime as dt, fcntl, hashlib, json, math, os, random, shutil
import subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import run_v25_simple_baselines as b
import v25_native_adapters as a
import v25_calendar as calendar

POLICY='native_fullgrid_mixed_ood_v1'

def freeze(root,suite):
    if suite.exists():raise FileExistsError(suite)
    suite.mkdir(parents=True);snap=suite/'source_snapshot';snap.mkdir()
    ref=root/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7'
    job=b.read(ref/'jobs/taxibj_I25_seed7.json')
    shutil.copytree(ref/'source_snapshot/src',snap/'src',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['run_v25_native_baselines.py','v25_native_adapters.py','v25_calendar.py','run_v25_simple_baselines.py']:
        shutil.copy2(Path(__file__).parent/name,snap/name)
    shutil.copy2(root/'tests/test_v25_native_adapters.py',snap/'test_v25_native_adapters.py')
    shutil.copy2(root/'imputation_benchmark/docs/V25_NATIVE_BASELINES.md',snap/'V25_NATIVE_BASELINES.md')
    provenance={}
    for repo in set(a.REPOS.values()):
        src=root/'imputation_benchmark/baselines'/repo;dst=snap/'upstream'/repo
        provenance[repo]={'commit':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),
            'origin':subprocess.check_output(['git','-C',str(src),'remote','get-url','origin'],text=True).strip()}
        for p in src.rglob('*'):
            rel=p.relative_to(src)
            if '.git' in rel.parts or '__pycache__' in rel.parts or not p.is_file():continue
            if p.suffix in ['.py','.yaml','.yml','.md','.txt'] or p.name.startswith('LICENSE'):
                d=dst/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for tag,sid in [('I25','2c07a1e3f7f177b7')]:
        d=suite/'references'/tag;d.mkdir(parents=True)
        shutil.copy2(ref/'evaluations'/f'taxibj_{tag}_seed7.json',d/'evaluations.json')
    calendar.prepare(root,suite/'data_bridge',job['sources'])
    import importlib.metadata as metadata
    deps=root/'imputation_benchmark/.native_baseline_deps'
    dependency_versions={d.metadata['Name']:d.version for d in metadata.distributions(path=[str(deps)])}
    m=dict(schema=1,policy=POLICY,baseline_policy=b.BASELINE_POLICY,root=str(root),
        created=dt.datetime.now().isoformat(),config=job['config'],sources=job['sources'],
        sources_sha256={k:b.sha(v) for k,v in job['sources'].items()},methods=a.METHODS,
        epochs=100,seed=7,validation_interval=5,batch_candidates=[32,16,8,4,2,1],
        train='TaxiBJ mixed four families 0.4; source train targets only; no M/C model inputs',
        protocol=job['config']['experiment_plan']['protocol'],optimizer='Adam',lr=.001,
        lr_schedule='cosine to .0003 over 100 epochs',precision='float32',
        native_sources=provenance,descriptions=a.DESCRIPTIONS,
        dependencies=str(root/'imputation_benchmark/.native_baseline_deps'),
        dependency_versions=dependency_versions,calendar=str(suite/'data_bridge/calendar.npz'),
        runtime_versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),
        calendar_sha256=b.sha(suite/'data_bridge/calendar.npz'),
        deferred={'SPIN':'已下载官方源码；待完成其图注意力依赖及位置输入适配审查。',
                  'MPIN':'已核实作者源码；数据流更新设定需要单独核实。'},
        hashes={str(p.relative_to(suite)):b.sha(p) for p in snap.rglob('*') if p.is_file()})
    b.write(suite/'manifest.json',m);print(suite,flush=True)

def setup(suite):
    m=b.read(suite/'manifest.json')
    if m['policy']!=POLICY:raise RuntimeError('Wrong suite policy')
    for p,h in m['hashes'].items():
        if b.sha(suite/p)!=h:raise RuntimeError('Changed frozen source '+p)
    for k,p in m['sources'].items():
        if b.sha(p)!=m['sources_sha256'][k]:raise RuntimeError('Changed data '+p)
    if b.sha(m['calendar'])!=m['calendar_sha256']:raise RuntimeError('Changed calendar mapping')
    sys.path[:0]=[m['dependencies'],str(suite/'source_snapshot/src')]
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return m

def seed(value=7):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)

def loader(ds,bs,shuffle=False,g=None):
    return torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=shuffle,generator=g,num_workers=0,drop_last=False)

def dataset(m,split,spec=None):
    return calendar.CalendarDataset(b.dataset(m,split,spec),m['calendar'],split)

@torch.no_grad()
def evaluate(m,suite,name,net,stats,split,bs,spec=None):
    ds=dataset(m,split,spec);scores=b.Scores();mh=hashlib.sha256();start=time.monotonic();seconds=0.
    net.eval()
    for batch in loader(ds,bs):
        # Each sample is hashed independently, so digest does not depend on batch size.
        for i in range(len(batch['m_f'])):
            mh.update(batch['m_f'][i].numpy().tobytes())
        tick=time.monotonic();pred,_=a.forward(name,net,batch,stats,'cuda:0');torch.cuda.synchronize()
        seconds+=time.monotonic()-tick;scores.update(pred,batch)
    return dict(raw=scores.result(),samples=len(ds),effective_mask_seed=ds.diverse_masks.seed,
        f_mask_sha256=mh.hexdigest(),seconds=time.monotonic()-start,forward_ms_per_window=1000*seconds/len(ds))

def preflight(m,suite,name,bs):
    from stmoe_imputer.utils.gpu_monitor import GPUMonitor
    seed(123);run=suite/'runs'/name
    with GPUMonitor(run):
        torch.cuda.set_per_process_memory_fraction(.88)
        stats=b.fit_statistics(m,suite)
        batch=next(iter(loader(dataset(m,'val'),bs)))
        net=a.build(name,suite/'source_snapshot/upstream',torch.device('cuda:0'))
        opt=torch.optim.Adam(net.parameters(),lr=.001);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        # Repeat an actual optimizer update to exercise allocated optimizer states too.
        for _ in range(2):
            net.train();opt.zero_grad(set_to_none=True)
            pred,loss=a.forward(name,net,batch,stats,'cuda:0',True)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        opt.zero_grad(set_to_none=True);net.eval()
        generator=torch.Generator().manual_seed(7)
        inference_rng=b.rng_state(generator)
        with torch.no_grad():expected,_=a.forward(name,net,batch,stats,'cuda:0')
        # Holdout/coarse pollution must not change inference.
        dirty={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
        dirty['x_f_gt'].fill_(float('nan'));dirty['x_f_obs'][dirty['m_f']==0]=float('nan')
        for k in ['x_m_gt','x_c_gt','x_m_obs','x_c_obs']:
            if k in dirty:dirty[k].fill_(float('nan'))
        b.restore_rng(inference_rng,generator)
        with torch.no_grad():actual,_=a.forward(name,net,dirty,stats,'cuda:0')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        cp=run/'preflight/restore.pth';b.save_checkpoint(cp,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=inference_rng))
        del pred,loss,actual,net,opt;torch.cuda.empty_cache()
        net=a.build(name,suite/'source_snapshot/upstream',torch.device('cuda:0'));opt=torch.optim.Adam(net.parameters(),lr=.001)
        saved=torch.load(cp,map_location='cpu',weights_only=False);net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);b.restore_rng(saved['rng'],generator);net.eval()
        with torch.no_grad():actual,_=a.forward(name,net,batch,stats,'cuda:0')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        del actual,expected,saved;net.train();opt.zero_grad(set_to_none=True)
        _,loss=a.forward(name,net,batch,stats,'cuda:0',True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        result=dict(status='passed',method=name,batch_size=bs,parameters=sum(p.numel() for p in net.parameters()),
            peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,
            checks=['fullgrid forward','backward and optimizer','hidden/coarse pollution','checkpoint forward parity','restored optimizer update'])
        b.write(run/'preflight/passed.json',result);print(json.dumps(result),flush=True)
        cp.unlink()

def train(m,suite,name,bs):
    from stmoe_imputer.utils.gpu_monitor import GPUMonitor
    seed();stats=b.fit_statistics(m,suite);run=suite/'runs'/name;g=torch.Generator().manual_seed(7)
    net=a.build(name,suite/'source_snapshot/upstream',torch.device('cuda:0'))
    opt=torch.optim.Adam(net.parameters(),lr=m['lr']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,99,eta_min=.0003)
    cp=run/'checkpoints/last.pth';begin=1;best=math.inf;best_epoch=0;elapsed=0.;peak=0.
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        if saved['policy']!=POLICY or saved['batch_size']!=bs:raise RuntimeError('Resume configuration mismatch')
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        b.restore_rng(saved['rng'],g);begin=saved['epoch']+1;best=saved['best'];best_epoch=saved['best_epoch'];elapsed=saved['elapsed'];peak=saved['peak_gib'];del saved
    with GPUMonitor(run):
        ds=dataset(m,'train')
        for epoch in range(begin,m['epochs']+1):
            ds.set_epoch(epoch);net.train();start=time.monotonic();total=0.;n=0;torch.cuda.reset_peak_memory_stats()
            for it,batch in enumerate(loader(ds,bs,True,g)):
                opt.zero_grad(set_to_none=True);pred,loss=a.forward(name,net,batch,stats,'cuda:0',True)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
                total+=loss.item()*len(pred);n+=len(pred)
                if it%20==0:
                    b.write(suite/'state.json',dict(phase='training',method=name,epoch=epoch,batch=it,batch_size=bs,seen=n))
                    print(f'{name} epoch {epoch}/100 batch {it} samples {n}/{len(ds)} loss {loss.item():.5f}',flush=True)
            row=dict(epoch=epoch,loss=total/n,lr=opt.param_groups[0]['lr'],batch_size=bs)
            if epoch%5==0:
                val=evaluate(m,suite,name,net,stats,'val',bs);row.update(val_mae=val['raw']['mae'],val_rmse=val['raw']['rmse'])
                if row['val_mae']<best:
                    best=row['val_mae'];best_epoch=epoch
                    b.save_checkpoint(run/'checkpoints/best.pth',dict(model=net.state_dict(),epoch=epoch,best=best,policy=POLICY))
            torch.cuda.synchronize();row['seconds']=time.monotonic()-start;elapsed+=row['seconds']
            peak=max(peak,torch.cuda.max_memory_allocated()/2**30);row['peak_gib']=peak
            if epoch<100:sched.step()
            b.save_checkpoint(cp,dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=b.rng_state(g),
                epoch=epoch,best=best,best_epoch=best_epoch,elapsed=elapsed,peak_gib=peak,batch_size=bs,policy=POLICY))
            b.log(run/'logs/train.jsonl',row);print(name,json.dumps(row),flush=True)
        saved=torch.load(run/'checkpoints/best.pth',map_location='cpu',weights_only=False);net.load_state_dict(saved['model']);net.eval()
        info=dict(best_epoch=best_epoch,best_val=best,train_seconds=elapsed,peak_gib=peak,batch_size=bs,
                  parameters=sum(p.numel() for p in net.parameters()))
        dest=suite/'results'/f'{name}.json';result=dict(method=name,status='evaluating',fit=info,policy=POLICY,sets={})
        for label,spec in m['protocol']['evaluations'].items():
            score=evaluate(m,suite,name,net,stats,'test',bs,spec)
            ref=b.read(suite/'references/I25/evaluations.json')['sets'][label]
            assert score['raw']['count']==ref['metrics']['metric_missing_count']
            assert score['effective_mask_seed']==ref['effective_mask_seed'] and score['samples']==ref['samples']
            for other in (suite/'results').glob('*.json'):
                prev=b.read(other).get('sets',{}).get(label)
                if prev:assert prev['f_mask_sha256']==score['f_mask_sha256']
            result['sets'][label]=score;b.write(dest,result);print(name,label,score['raw'],flush=True)
        result['status']='finished';b.write(dest,result)


def report(m,suite):
    rows={n:b.read(suite/'results'/f'{n}.json') for n in m['methods'] if (suite/'results'/f'{n}.json').exists()}
    finished=all(rows.get(n,{}).get('status')=='finished' or (suite/'runs'/n/'failed.json').exists() for n in m['methods'])
    kind='完整' if finished else '阶段';date=dt.datetime.now().strftime('%Y%m%d')
    path=Path(m['root'])/'experments_report'/f'{date}_v25_原生补全基线混合缺失泛化_{suite.name}_{kind}分析报告.md'
    lines=[f'# 原生补全基线混合缺失泛化：{kind}分析报告','',f'套件：`{suite}`。','',
        '## 1. 具体方法与复现配置','','|方法|具体做法|','|---|---|']
    lines += [f'|{n}|{m["descriptions"][n]}|' for n in m['methods']]
    lines += ['', 'F-only；不加入M/C观测、投影或本课题的辅助损失。两个原始流量通道都评估，序列模型展开全网格为2048特征；图模型与ImputeFormer把两通道视为两个独立1024节点样本，共享模型。窗口不拼接，4邻接只由坐标构建。',
        'TaxiBJ四类混合mask0.4、seed7、100epoch、每5轮ID验证选best。Adam 1e-3余弦至3e-4、clip5、FP32、无早停。batch依原尺寸GPU预检按32/16/8/4/2/1回退；实际batch及原生监督差异均披露，不声称复现论文最佳结果。统计量只用训练可见F。',
        'SAITS保持ORT+MIT；T1用缺失MSE；BRITS/RITS/MRNN保持原生观测重建损失；GRIN含原生中间预测损失与5%观测隐藏；MPGRU/BiMPGRU保持原生观测重建。ImputeFormer保持0.01原生Fourier项；源训练数据完整，synthetic original_mask=1，Freg只读取预测，避免将隐藏值作为可见观测。其官方FFT涵盖batch维，故batch改变也会影响原生正则项，这一限制保留并披露。',
        'GRIN四个中间预测L1各权重1相加；PAST保留原生观测MSE与detach残差辅助MSE；STAM保留原生物理量空间非零缺失MAE，正式评估仍包含零值。PAST/STAM使用TAXIBJ.ext与清洗frame_ids核验恢复的真实日历，保留Z时间戳时钟、不添加天气；详见冻结calendar_audit.json。STAM保留eval随机采样行为，预检的污染与恢复对照固定相同RNG。',
        '六套测试使用同一best，不重新拟合；按缺失元素计算MAE/RMSE，记录各缺失族和有效计数。二元组合及未见形态各自先平均两次复测，再与三元组合等权汇总。',
        f'冻结源码、原仓库commit、配置和数据SHA256：`{suite}/manifest.json`；原模型未改动，仅由独立桥接器调用。GPU1单卡串行，监控每15秒。',
        f'复现：`CUDA_VISIBLE_DEVICES=1 {sys.executable} {suite}/source_snapshot/run_v25_native_baselines.py --suite {suite}`。',
        '', '## 2. 完整结果与诊断','','|方法|状态|batch|参数|best epoch|ID val MAE|训练分钟|峰值GiB|','|---|---|---|---|---|---|---|---|']
    for n in m['methods']:
        r=rows.get(n);p=suite/'runs'/n/'preflight/passed.json';failure=suite/'runs'/n/'failed.json'
        if r:
            i=r['fit'];lines.append(f'|{n}|{r["status"]}|{i["batch_size"]}|{i["parameters"]}|{i["best_epoch"]}|{i["best_val"]:.4f}|{i["train_seconds"]/60:.2f}|{i["peak_gib"]:.3f}|')
        else:lines.append(f'|{n}|'+('失败' if failure.exists() else '预检通过/待训练或训练中' if p.exists() else '待预检')+'|—|—|—|—|—|—|')
    lines+=['','|方法|协议|MAE|RMSE|缺失计数|毫秒/窗口|','|---|---|---|---|---|---|']
    for n,r in rows.items():
        for label,s in r['sets'].items():
            v=s['raw'];lines.append(f'|{n}|{label}|{v["mae"]:.4f}|{v["rmse"]:.4f}|{v["count"]}|{s["forward_ms_per_window"]:.2f}|')
    refs=[('I25 FMC参考',suite/'references/I25/evaluations.json'),('I29 F-only',Path(m['root'])/'outputs/v25/experiments/taxibj_input_communication/0b30aec06fafb7ca/evaluations/taxibj_I29_seed7.json')]
    for label,p in refs:
        if p.exists():
            for protocol,s in b.read(p)['sets'].items():
                v=s['metrics'];lines.append(f'|{label}|{protocol}|{v["mae"]:.4f}|{v["rmse"]:.4f}|{int(v["metric_missing_count"])}|—|')
    lines+=['','### 三类OOD等权汇总','','|方法|ID MAE|二元MAE|形态MAE|三元MAE|三类平均MAE|三类平均RMSE|','|---|---|---|---|---|---|---|']
    groups=[['unseen_combinations','unseen_combinations_repeat'],['unseen_geometry','unseen_geometry_repeat'],['unseen_triple']]
    summaries=[]
    combined={n:{k:v['raw'] for k,v in r['sets'].items()} for n,r in rows.items()}
    for label,p in refs:
        if p.exists():combined[label]={k:v['metrics'] for k,v in b.read(p)['sets'].items()}
    for n,sets in combined.items():
        if not all(k in sets for group in groups for k in group) or 'in_distribution' not in sets:continue
        means=[float(np.mean([sets[k]['mae'] for k in group])) for group in groups]
        rmses=[float(np.mean([sets[k]['rmse'] for k in group])) for group in groups]
        item=dict(method=n,id=sets['in_distribution']['mae'],ood=float(np.mean(means)),rmse=float(np.mean(rmses)))
        summaries.append(item)
        lines.append(f'|{n}|{item["id"]:.4f}|{means[0]:.4f}|{means[1]:.4f}|{means[2]:.4f}|{item["ood"]:.4f}|{item["rmse"]:.4f}|')
    lines+=['','以上为类别内、类别间的算术平均；不是把所有元素合并重新计算的RMSE。']
    lines+=['','### 训练验证曲线','','|方法|epoch|训练loss|ID val MAE|','|---|---|---|---|']
    for n in m['methods']:
        p=suite/'runs'/n/'logs/train.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if 'val_mae' in r:lines.append(f'|{n}|{r["epoch"]}|{r["loss"]:.6f}|{r["val_mae"]:.4f}|')
    lines+=['','### 缺失族与失败披露','','|方法|协议|缺失族|MAE|RMSE|计数|','|---|---|---|---|---|---|']
    for n,r in rows.items():
        for label,s in r['sets'].items():
            for f,v in s['raw']['families'].items():lines.append(f'|{n}|{label}|{f}|{v["mae"]:.4f}|{v["rmse"]:.4f}|{v["count"]}|')
    for p in (suite/'runs').glob('*/failed.json'):lines+=['',f'失败：`{p}`', '```json',p.read_text(),'```']
    lines+=['','## 3. 分析与证据边界','',
        '单seed探索；mask复测不是训练种子复验。I29同为F-only，是输入信息一致的主参照；I25有额外多尺度观测，只能作为不同信息条件的参考。',
        '本批主动将各模型放到统一组合缺失/OOD协议，不能据此称各原论文已经采用相同协议。不同原生损失、通道组织、batch及配置均可能影响结果。',
        '已核实T1有未见缺失模式实验、ImputeFormer有跨缺失率零样本实验；STAMImputer的块缺失叠加随机缺失与本课题的未见组合协议不同。',
        '具体候选来源与暂缓原因见 `imputation_benchmark/docs/V25_NATIVE_BASELINES.md`。所有未完成或失败模型保留在上表，不以精度筛掉负结果。']
    compared=next((s for s in summaries if s['method']=='I29 F-only'),None)
    if compared:
        for item in summaries:
            if item['method'] not in m['methods']:continue
            lines+=['',f'{item["method"]} 相对 I29：ID MAE变化 {(item["id"]/compared["id"]-1)*100:+.2f}%，三类等权 OOD MAE变化 {(item["ood"]/compared["ood"]-1)*100:+.2f}%（负值为更好）。此为本预算下的实测比较，不足以归因于单个模型机制。']
    lines+=['','下一步：先核查验证曲线、各缺失族与成本，再对有竞争力的模型进行训练种子复验；若未胜过现有基线，完整保留负结果。']
    path.write_text('\n'.join(lines)+'\n');b.write(suite/'report_path.json',dict(path=str(path),complete=finished));return path,finished


def gpu_free():
    out=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=8)
    return not out.strip()

def queue(m,suite):
    lockpath=Path(m['root'])/'outputs/v24-COE/experiments/gpu1.queue.lock'
    with (suite/'queue.lock').open('a') as own,lockpath.open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                b.write(suite/'state.json',dict(phase='waiting_for_existing_gpu1_queue'));print('Waiting for existing GPU1 queue',flush=True);time.sleep(15)
        while not gpu_free():
            print('Waiting for other GPU1 compute processes',flush=True);time.sleep(15)
        for name in m['methods']:
            dest=suite/'results'/f'{name}.json'
            if dest.exists() and b.read(dest)['status']=='finished':continue
            run=suite/'runs'/name;(run/'logs').mkdir(parents=True,exist_ok=True)
            passed=run/'preflight/passed.json'
            if not passed.exists():
                for bs in m['batch_candidates']:
                    b.write(suite/'state.json',dict(phase='preflight',method=name,batch_size=bs))
                    cmd=[sys.executable,str(suite/'source_snapshot/run_v25_native_baselines.py'),'--suite',str(suite),'--method',name,'--batch-size',str(bs),'--worker','preflight']
                    with (run/'logs'/f'preflight_bs{bs}.log').open('a') as f:
                        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                    print(f'{name} preflight bs={bs} exit={p.returncode}',flush=True)
                    if p.returncode==0:break
                    if p.returncode!=77:
                        b.write(run/'failed.json',dict(phase='preflight',batch_size=bs,exit=p.returncode));report(m,suite)
                        raise RuntimeError(f'{name}: non-memory preflight failure; inspect log')
                if not passed.exists():
                    b.write(run/'failed.json',dict(phase='preflight',reason='OOM through batch1'));report(m,suite);continue
            bs=b.read(passed)['batch_size']
            cmd=[sys.executable,str(suite/'source_snapshot/run_v25_native_baselines.py'),'--suite',str(suite),'--method',name,'--batch-size',str(bs),'--worker','train']
            # Stream output simultaneously to dedicated tmux and persistent per-run log.
            with (run/'logs/console.log').open('a') as f:
                p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
                code=p.wait()
            if code:
                b.write(run/'failed.json',dict(phase='train_or_eval',exit=code));report(m,suite);raise RuntimeError(f'{name} failed; preserved checkpoint, no automatic retry')
            report(m,suite)
        path,finished=report(m,suite)
        success=all((suite/'results'/f'{n}.json').exists() and b.read(suite/'results'/f'{n}.json')['status']=='finished' for n in m['methods'])
        b.write(suite/'queue_finished.json',dict(all_successful=success,report=str(path)))
        b.write(suite/'state.json',dict(phase='completed' if success else 'completed_with_skips',report=str(path)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--root',type=Path)
    p.add_argument('--method',choices=a.METHODS);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--worker',choices=['preflight','train']);p.add_argument('--report-only',action='store_true');arg=p.parse_args();suite=arg.suite.resolve()
    if arg.prepare:freeze(arg.root.resolve(),suite);return
    m=setup(suite)
    if arg.report_only:print(report(m,suite));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('GPU1 only')
    try:
        if arg.worker=='preflight':preflight(m,suite,arg.method,arg.batch_size)
        elif arg.worker=='train':train(m,suite,arg.method,arg.batch_size)
        else:queue(m,suite)
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc();sys.exit(77)
    except BaseException:
        if not arg.worker:b.write(suite/'failed.json',dict(time=dt.datetime.now().isoformat(),error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
