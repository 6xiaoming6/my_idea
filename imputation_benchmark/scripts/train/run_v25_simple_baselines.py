#!/usr/bin/env python3
"""Small baseline suite, with immutable v25 masks and unmodified upstream RNNs.

No model registration changes. Statistics use training observations only. Neural
models see nonoverlapping spatial patches, never concatenated time windows.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from scipy.ndimage import uniform_filter

METHODS = ['ChannelMean', 'NodeMean', 'TemporalLinear', 'SpatialMean', 'SoftImpute', 'RITS_patch8', 'BRITS_patch8']
NEURAL = METHODS[-2:]
BASELINE_POLICY = 'f_only_native_loss_no_projection_v2'
DESCRIPTIONS = {
    'ChannelMean': '训练可见F按通道求均值；所有缺失位置填对应通道均值。',
    'NodeMean': '训练可见F按通道及网格位置求均值，跨训练窗口与时间平均；不是日历周期均值。',
    'TemporalLinear': '每个窗口内各通道/网格时间线性插值，边界取最近可见值；整段缺失回退NodeMean。',
    'SpatialMean': '同一时间片可见邻域均值，窗口3/5/9由ID验证MAE选择；空邻域回退NodeMean。',
    'SoftImpute': '各窗口独立T×(C·H·W)矩阵，减训练NodeMean后软阈值SVD迭代；无跨窗口拼接。',
    'RITS_patch8': 'SAITS仓库提供的未修改RITS实现；8×8非重叠空间块，保留12步及两通道，hidden32。',
    'BRITS_patch8': '同一仓库未修改BRITS实现；与RITS相同数据适配，双向循环与原一致性项。',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save_checkpoint(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(data, tmp); tmp.replace(path)


def log(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f: f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(root, suite):
    if (suite / 'manifest.json').exists():
        raise FileExistsError('Frozen suite already exists; run it without --prepare')
    ref = root / 'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7'
    job = read(ref / 'jobs/taxibj_I25_seed7.json')
    suite.mkdir(parents=True, exist_ok=True)
    snap = suite / 'source_snapshot'
    shutil.copytree(ref / 'source_snapshot/src', snap / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    up = root / 'imputation_benchmark/baselines/SAITS'
    (snap / 'upstream/modeling').mkdir(parents=True)
    for name in ['brits.py', 'utils.py']:
        shutil.copy2(up / 'modeling' / name, snap / 'upstream/modeling' / name)
    shutil.copy2(up / 'LICENSE', snap / 'upstream/LICENSE')
    shutil.copy2(__file__, snap / 'run_v25_simple_baselines.py')
    shutil.copy2(root / 'tests/test_v25_simple_baselines.py', snap / 'test_v25_simple_baselines.py')
    for name, sid in [('I25', '2c07a1e3f7f177b7'), ('I27', '6df045b55c2886df')]:
        b = root / 'outputs/v25/experiments/taxibj_input_communication' / sid
        for folder in ['configs', 'results', 'evaluations']:
            dest = suite / 'references' / name / (folder + '.json')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b / folder / f'taxibj_{name}_seed7.json', dest)
    manifest = dict(schema=2, baseline_policy=BASELINE_POLICY, root=str(root), created=dt.datetime.now().isoformat(),
                    config=job['config'], sources=job['sources'], methods=METHODS,
                    epochs=100, batch_size=32, seed=7, patch=8, hidden=32,
                    lr=1e-3, eta_min=3e-4, weight_decay=1e-4, amp=False,
                    source_note='RITS/BRITS from vendored WenjieDu/SAITS modeling/brits.py; source unmodified, not SAITS attention model',
                    projection='none', training_loss='native reconstruction plus native bidirectional consistency; MIT disabled',
                    sources_sha256={k:sha(v) for k,v in job['sources'].items()},
                    hashes={str(p.relative_to(suite)):sha(p) for p in snap.rglob('*') if p.is_file()})
    write(suite / 'manifest.json', manifest)
    print('Frozen suite:', suite, flush=True)


def validate_policy(m):
    if m.get('baseline_policy') != BASELINE_POLICY:
        raise RuntimeError('Legacy suite used extra losses/projection. Keep its frozen artifacts; prepare a new suite for native baselines.')


def setup(suite):
    m = read(suite / 'manifest.json')
    validate_policy(m)
    for rel, expected in m['hashes'].items():
        if sha(suite / rel) != expected: raise RuntimeError('Frozen source changed: ' + rel)
    for key, path in m['sources'].items():
        if sha(path) != m['sources_sha256'][key]: raise RuntimeError('Data changed: ' + key)
    sys.path[:0] = [str(suite / 'source_snapshot/src'), str(suite / 'source_snapshot/upstream')]
    torch.set_num_threads(2)
    return m


def dataset(m, split, spec=None):
    from stmoe_imputer.data.observed_multiscale import build_observed_dataset
    cfg = copy.deepcopy(m['config'])
    if spec:
        cfg['data']['eval_mask_diversity'] = dict(families=spec['families'], rates=[0.4], seed=spec['seed'], resample_each_epoch=False)
    return build_observed_dataset(cfg, m['sources'][split], split)


def loader(ds, shuffle=False, generator=None):
    return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0,
                                       generator=generator, drop_last=False)


def fit_statistics(m, suite):
    path = suite / 'statistics.npz'
    if path.exists():
        with np.load(path) as z: return {k:z[k] for k in z.files}
    ds = dataset(m, 'train'); ds.set_epoch(1)
    sums = counts = squares = None
    for b in loader(ds):
        x = b['x_f_obs'].numpy().astype(np.float64); mask = b['m_f'].numpy()
        s, c, q = x.sum((0,2)), mask.sum((0,2)), (x*x).sum((0,2))
        sums = s if sums is None else sums+s
        counts = c if counts is None else counts+c
        squares = q if squares is None else squares+q
    channel = sums.sum((1,2)) / counts.sum((1,2)).clip(1)
    node = np.where(counts > 0, sums / counts.clip(1), channel[:,None,None])
    std = np.sqrt(np.maximum(squares / counts.clip(1) - node**2, 1.0))
    stats = dict(channel=channel.astype('float32'), node=node.astype('float32'), std=std.astype('float32'), counts=counts)
    np.savez(path, **stats)
    write(suite / 'statistics_provenance.json', dict(split='train', epoch=1, visible_only=True, mask_seed=20260917,
          samples=len(ds), sha256=sha(path), missing_nodes=int((counts==0).sum())))
    return stats


def temporal(x, mask, prior):
    # Vectorized nearest valid left/right times. No timeline crosses a window.
    n,c,t,h,w = x.shape
    rows = x.transpose(0,1,3,4,2).reshape(-1,t)
    vis = mask.transpose(0,1,3,4,2).reshape(-1,t).astype(bool)
    grid = np.arange(t)[None]
    left = np.maximum.accumulate(np.where(vis, grid, -1), axis=1)
    right = np.minimum.accumulate(np.where(vis, grid, t)[:,::-1], axis=1)[:,::-1]
    lv = np.take_along_axis(rows, left.clip(0,t-1), 1)
    rv = np.take_along_axis(rows, right.clip(0,t-1), 1)
    ratio = (grid-left) / np.maximum(right-left,1)
    result = np.where(left<0, rv, np.where(right>=t, lv, lv+(rv-lv)*ratio))
    fallback = np.broadcast_to(prior[None], (n,c,h,w)).reshape(-1,1)
    result = np.where(vis.any(1)[:,None], result, fallback)
    return result.reshape(n,c,h,w,t).transpose(0,1,4,2,3).astype('float32')


def traditional(name, batch, stats, option):
    # Deliberately access observations only, not any *_gt field.
    mask = batch['m_f'].numpy().astype(bool)
    x = np.where(mask, batch['x_f_obs'].numpy(), 0.)
    prior = np.broadcast_to(stats['node'][None,:,None], x.shape)
    if name == 'ChannelMean': pred = np.broadcast_to(stats['channel'][None,:,None,None,None], x.shape)
    elif name == 'NodeMean': pred = prior
    elif name == 'TemporalLinear': pred = temporal(x, mask, stats['node'])
    elif name == 'SpatialMean':
        size = (1,1,1,option,option)
        num = uniform_filter(x, size=size, mode='constant')
        den = uniform_filter(mask.astype('float32'), size=size, mode='constant')
        pred = np.where(den>1e-8, num / np.maximum(den,1e-8), prior)
    elif name == 'SoftImpute':
        n,c,t,h,w = x.shape
        obs = mask.transpose(0,2,1,3,4).reshape(n,t,-1)
        base = prior.transpose(0,2,1,3,4).reshape(n,t,-1)
        residual = np.where(obs, x.transpose(0,2,1,3,4).reshape(n,t,-1)-base, 0.).astype('float64')
        z = np.zeros_like(residual)
        # Per-window regularization depends only on its observed residuals.
        lam = np.linalg.svd(residual, compute_uv=False)[:,0] * option
        for _ in range(50):
            filled = np.where(obs, residual, z)
            u,s,v = np.linalg.svd(filled, full_matrices=False)
            new = (u * np.maximum(s-lam[:,None],0)[:,None,:]) @ v
            change = np.linalg.norm(new-z,axis=(1,2)) / np.maximum(np.linalg.norm(z,axis=(1,2)),1e-8)
            z = new
            if np.max(change) < 1e-4: break
        pred = (z+base).reshape(n,t,c,h,w).transpose(0,2,1,3,4)
    else: raise ValueError(name)
    return torch.from_numpy(np.where(mask,x,pred).astype('float32'))


def patches(x, p=8):
    b,c,t,h,w = x.shape
    return x.reshape(b,c,t,h//p,p,w//p,p).permute(0,3,5,2,1,4,6).reshape(b*(h//p)*(w//p),t,c*p*p)


def unpatch(x, shape, p=8):
    b,c,t,h,w = shape
    return x.reshape(b,h//p,w//p,t,c,p,p).permute(0,4,3,1,5,2,6).reshape(shape)


def gaps(mask):
    # Same convention as upstream unified_dataloader: delta_t=1+(1-mask_t)*delta_(t-1).
    out = torch.zeros_like(mask)
    for i in range(1,mask.shape[1]): out[:,i] = 1+(1-mask[:,i])*out[:,i-1]
    return out


def neural_inputs(batch, stats, device):
    x = batch['x_f_obs'].to(device); mask = batch['m_f'].to(device)
    mean = torch.as_tensor(stats['node'],device=device)[None,:,None]
    std = torch.as_tensor(stats['std'],device=device)[None,:,None]
    values = patches(torch.where(mask.bool(), (x-mean)/std, 0.))
    masks = patches(mask)
    inputs = {'forward':dict(X=values, missing_mask=masks, deltas=gaps(masks))}
    rev = masks.flip(1)
    inputs['backward'] = dict(X=values.flip(1), missing_mask=rev, deltas=gaps(rev))
    return inputs, mean, std


def new_model(name, device):
    from modeling.brits import RITS, BRITS
    cls = BRITS if name.startswith('BRITS') else RITS
    return cls(seq_len=12, feature_num=128, rnn_hidden_size=32, device=str(device), MIT=False).to(device)


def neural_forward(model, name, batch, stats, device, training=False):
    inputs, mean, std = neural_inputs(batch, stats, device)
    # Targets are kept outside the model forward; never put holdout into inputs.
    out = model(inputs, 'train' if training else 'test') if name.startswith('BRITS') else model(inputs)
    pred = unpatch(out['imputed_data'], batch['x_f_obs'].shape) * std + mean
    pred = torch.where(batch['m_f'].to(device).bool(), batch['x_f_obs'].to(device), pred)
    loss = None
    if training:
        loss = out['reconstruction_loss'] + out['consistency_loss']
    return pred, loss


class Scores:
    def __init__(self): self.count=0; self.ae=0.; self.se=0.; self.family={}
    def update(self, pred, batch):
        from stmoe_imputer.data.diverse_masks import ALL_FAMILIES
        pred=pred.detach().cpu().double(); target=batch['x_f_gt'].double(); q=batch['target_mask'].bool()
        if not torch.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
        err=pred-target
        a=(err.abs()*q).sum((1,2,3,4)); s=(err.square()*q).sum((1,2,3,4)); n=q.sum((1,2,3,4))
        self.ae+=a.sum().item();self.se+=s.sum().item();self.count+=n.sum().item()
        for family in batch['mask_family'].unique():
            pick=batch['mask_family']==family; key=ALL_FAMILIES[int(family)]
            row=self.family.setdefault(key,[0.,0.,0]);row[0]+=a[pick].sum().item();row[1]+=s[pick].sum().item();row[2]+=n[pick].sum().item()
    def result(self):
        return dict(mae=self.ae/self.count, rmse=math.sqrt(self.se/self.count), count=self.count,
                    families={k:dict(mae=a/n,rmse=math.sqrt(s/n),count=n) for k,(a,s,n) in self.family.items()})


@torch.no_grad()
def evaluate(ds, predict):
    raw=Scores(); start=time.monotonic(); maskhash=hashlib.sha256(); forward_seconds=0.
    for batch in loader(ds):
        for k in ['m_f','m_m','m_c']: maskhash.update(batch[k].numpy().tobytes())
        tick=time.monotonic(); pred=predict(batch)
        if pred.is_cuda: torch.cuda.synchronize()
        forward_seconds+=time.monotonic()-tick
        raw.update(pred,batch)
    return dict(raw=raw.result(),
                seconds=time.monotonic()-start, forward_ms_per_window=1000*forward_seconds/len(ds),
                samples=len(ds), mask_sha256=maskhash.hexdigest(), effective_mask_seed=ds.diverse_masks.seed)


def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),loader=generator.get_state())


def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda']);generator.set_state(state['loader'])


def train_neural(m,suite,name,stats,device):
    from stmoe_imputer.utils.gpu_monitor import GPUMonitor
    run=suite/'runs'/name; (run/'checkpoints').mkdir(parents=True,exist_ok=True)
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    generator=torch.Generator().manual_seed(7)
    model=new_model(name,device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:(.0003+.5*(.001-.0003)*(1+math.cos(math.pi*min(step,99)/99)))/.001)
    start_epoch=1;best=math.inf;best_epoch=0;elapsed=0.;peak=0.
    cp=run/'checkpoints/last.pth'
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        validate_policy(saved)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'],generator);start_epoch=saved['epoch']+1;best=saved['best_val'];best_epoch=saved['best_epoch'];elapsed=saved['elapsed'];peak=saved['peak_gib']
        del saved
    train=dataset(m,'train');val=dataset(m,'val')
    with GPUMonitor(run):
        for epoch in range(start_epoch,m['epochs']+1):
            train.set_epoch(epoch);model.train();torch.cuda.reset_peak_memory_stats();started=time.monotonic();sums=0.;n=0
            for batch in loader(train,True,generator):
                opt.zero_grad(set_to_none=True)
                pred,loss=neural_forward(model,name,batch,stats,device,True)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                sums+=loss.item()*len(pred);n+=len(pred)
            row=dict(epoch=epoch,loss=sums/n,lr=opt.param_groups[0]['lr'])
            if epoch%5==0:
                model.eval();scores=evaluate(val,lambda b:neural_forward(model,name,b,stats,device)[0])
                row['val_mae']=scores['raw']['mae'];row['val_rmse']=scores['raw']['rmse']
                if row['val_mae']<best:
                    best=row['val_mae'];best_epoch=epoch
                    save_checkpoint(run/'checkpoints/best.pth',dict(model=model.state_dict(),epoch=epoch,best_val=best,method=name,baseline_policy=BASELINE_POLICY))
            torch.cuda.synchronize();row['seconds']=time.monotonic()-started;elapsed+=row['seconds']
            peak=max(peak,torch.cuda.max_memory_allocated()/1024**3);row['peak_gib']=peak
            sched.step()
            save_checkpoint(cp,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                rng=rng_state(generator),epoch=epoch,best_val=best,best_epoch=best_epoch,elapsed=elapsed,peak_gib=peak,method=name,baseline_policy=BASELINE_POLICY))
            log(run/'logs/train.jsonl',row);write(suite/'state.json',dict(phase='training',method=name,**row))
            print(name,json.dumps(row),flush=True)
    saved=torch.load(run/'checkpoints/best.pth',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval()
    info=dict(epochs=100,best_epoch=best_epoch,best_val=best,train_seconds=elapsed,peak_gib=peak,
              parameters=sum(p.numel() for p in model.parameters()),option=None)
    return model,info


def preflight(m,suite,stats):
    # Full batch, full 32x32 windows; patch adapter affects only feature layout.
    batch=next(iter(loader(dataset(m,'val'))));device=torch.device('cuda:0');results={}
    for name in NEURAL:
        torch.manual_seed(123);model=new_model(name,device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        pred,loss=neural_forward(model,name,batch,stats,device,True);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(): expected=neural_forward(model,name,batch,stats,device)[0]
        path=suite/'preflight'/f'{name}.pth';save_checkpoint(path,dict(model=model.state_dict(),optimizer=opt.state_dict()))
        model2=new_model(name,device);opt2=torch.optim.AdamW(model2.parameters(),lr=.001)
        saved=torch.load(path,map_location='cpu',weights_only=False);model2.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer']);model2.eval()
        with torch.no_grad():actual=neural_forward(model2,name,batch,stats,device)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Complete optimizer restore: next deterministic update must also agree.
        for net,optim in [(model,opt),(model2,opt2)]:
            net.train();optim.zero_grad(set_to_none=True);_,l=neural_forward(net,name,batch,stats,device,True);l.backward();optim.step()
        for a,b in zip(model.parameters(),model2.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1024**3
        if peak>4:raise RuntimeError(f'{name} preflight exceeded 4 GiB guard: {peak}')
        results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),peak_gib=peak,seconds=time.monotonic()-start,
                           batch_windows=32,patch_sequences=512,forward_backward_optimizer_resume=True)
        del model,model2,opt,opt2,saved,pred,loss,expected,actual;torch.cuda.empty_cache()
    write(suite/'preflight/results.json',results);print('Preflight:',json.dumps(results),flush=True)


def report(m,suite,complete=False):
    validate_policy(m)
    date=dt.datetime.now().strftime('%Y%m%d');kind='完整' if complete else '阶段'
    path=Path(m['root'])/'experments_report'/f'{date}_v25_原生简单基线_{suite.name}_{kind}分析报告.md'
    lines=[f'# v25 原生简单基线：{kind}分析报告','',f'冻结套件：`{suite}`。',
      '', '## 1. 具体方法与复现配置','', '|方法|具体做法|','|---|---|']
    lines += [f'|{n}|{DESCRIPTIONS[n]}|' for n in METHODS]
    lines += ['', '仅做数据适配：全部基线只读取F观测，不添加M/C投影、融合模块或额外监督损失。RITS/BRITS使用未修改的本地SAITS仓库实现（不是SAITS模型），关闭MIT，仅使用其原生观测重建及原生双向一致性损失。',
      '', 'TaxiBJ、两通道、12步、原时间切分、混合mask0.4，复用I25数据与六套评估协议。训练mask逐epoch变化，统计量仅拟合训练epoch1可见F；误差按F缺失元素累计。',
      '空间均值窗口3/5/9；SoftImpute采用NodeMean中心化的12×2048窗口矩阵，奇异值相对阈值0.01/0.05/0.1/0.2，最多50次、容差1e-4。仅按ID验证MAE选超参。',
      'RNN将网格适配为16个8×8非重叠块，每块12×128，hidden32，训练可见F逐网格标准化。seed7、100epoch、batch32完整窗口、FP32、AdamW、1e-3余弦至3e-4、weight_decay1e-4、clip1、每5轮ID验证、无早停；仅按ID验证MAE选择best。GPU1单卡串行，每15秒监控。',
      f'复现：`CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 {sys.executable} {suite}/source_snapshot/run_v25_simple_baselines.py --suite {suite}`。源码及数据哈希见manifest.json；日志和检查点见runs；逐协议原始指标见results。',
      '', '## 2. 完整结果与诊断','', '|方法|状态|best轮次|ID验证MAE|参数量|训练分钟|峰值已分配GiB|','|---|---|---|---|---|---|---|']
    finished={}
    for name in METHODS:
        p=suite/'results'/f'{name}.json'
        if p.exists():
            r=read(p);validate_policy(r);finished[name]=r;i=r['fit']
            lines.append(f"|{name}|{r['status']}|{i.get('best_epoch','—')}|{i['best_val']:.4f}|{i.get('parameters',0)}|{i.get('train_seconds',0)/60:.2f}|{i.get('peak_gib','—')}|")
        else:lines.append(f'|{name}|未完成|—|—|—|—|—|')
    lines+=['','|方法|协议|MAE|RMSE|有效缺失元素|','|---|---|---|---|---|']
    for name,r in finished.items():
        for protocol,row in r['sets'].items():
            v=row['raw'];lines.append(f"|{name}|{protocol}|{v['mae']:.4f}|{v['rmse']:.4f}|{v['count']}|")
    for ref in ['I25','I27']:
        for protocol,row in read(suite/'references'/ref/'evaluations.json')['sets'].items():
            v=row['metrics'];lines.append(f"|{ref}（FMC参考）|{protocol}|{v['mae']:.4f}|{v['rmse']:.4f}|{int(v['metric_missing_count'])}|")
    lines+=['','### 各缺失族与推理成本','', '|方法|协议|缺失族|MAE|RMSE|计数|','|---|---|---|---|---|---|']
    for name,r in finished.items():
        for protocol,row in r['sets'].items():
            for fam,v in row['raw']['families'].items():lines.append(f"|{name}|{protocol}|{fam}|{v['mae']:.4f}|{v['rmse']:.4f}|{v['count']}|")
    lines+=['','|方法|协议|毫秒/窗口|评估秒|','|---|---|---|---|']
    for name,r in finished.items():
        for protocol,row in r['sets'].items():lines.append(f"|{name}|{protocol}|{row['forward_ms_per_window']:.3f}|{row['seconds']:.2f}|")
    lines+=['','### 训练曲线记录','', '|方法|epoch|训练损失|ID验证MAE|','|---|---|---|---|']
    for name in NEURAL:
        p=suite/'runs'/name/'logs/train.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if 'val_mae' in r:lines.append(f"|{name}|{r['epoch']}|{r['loss']:.6f}|{r['val_mae']:.4f}|")
    lines+=['','## 3. 分析与边界','',
      'F-only基线与I25/I27的FMC输入信息量不同，不能将全部精度差异归因于模型结构。RNN分块缺少跨块信息，不代表全网格或论文最佳配置。仅保留原生损失并不等于完全复现论文实验设置。',
      '本批为单seed，测试mask复测不视为训练种子。CPU与GPU推理耗时不能用于严格硬件成本排名；峰值已分配显存不等于nvidia-smi总占用。传统法没有训练曲线。',
      '后续以原生基线及同骨干消融分别验证整体结果与模块贡献；不通过新增模块提高或改变基线。']
    failure=suite/'failed.json'
    if failure.exists():lines+=['','失败记录：', '```json',json.dumps(read(failure),ensure_ascii=False,indent=2),'```']
    path.write_text('\n'.join(lines)+'\n');write(suite/'report_path.json',dict(path=str(path),complete=complete))
    return path


def run(m,suite,only_preflight=False):
    stats=fit_statistics(m,suite)
    if only_preflight:
        from stmoe_imputer.utils.gpu_monitor import GPUMonitor
        with GPUMonitor(suite/'preflight'): preflight(m,suite,stats)
        return
    if not (suite/'preflight/results.json').exists():raise RuntimeError('GPU preflight required')
    protocol=m['config']['experiment_plan']['protocol']
    for name in METHODS:
        dest=suite/'results'/f'{name}.json'
        if dest.exists():
            previous_result=read(dest);validate_policy(previous_result)
            if previous_result['status']=='finished':continue
        write(suite/'state.json',dict(phase='fitting',method=name))
        device=torch.device('cuda:0') if name in NEURAL else torch.device('cpu')
        if name in NEURAL:
            model,info=train_neural(m,suite,name,stats,device)
            predict=lambda b:neural_forward(model,name,b,stats,device)[0]
        else:
            candidates=[3,5,9] if name=='SpatialMean' else [.01,.05,.1,.2] if name=='SoftImpute' else [None]
            vals=[];start=time.monotonic();val=dataset(m,'val')
            for option in candidates:
                score=evaluate(val,lambda b:traditional(name,b,stats,option))
                vals.append(dict(option=option,**score));print(name,'val',option,score['raw']['mae'],flush=True)
            winner=min(vals,key=lambda r:r['raw']['mae']);option=winner['option']
            info=dict(option=option,best_val=winner['raw']['mae'],train_seconds=time.monotonic()-start,parameters=0)
            write(suite/'runs'/name/'validation.json',vals)
            predict=lambda b:traditional(name,b,stats,option)
        result=dict(status='evaluating',method=name,baseline_policy=BASELINE_POLICY,fit=info,sets={})
        for label,spec in protocol['evaluations'].items():
            ds=dataset(m,'test',spec);scores=evaluate(ds,predict)
            # A frozen reference count check catches channel/split/mask errors.
            ref=read(suite/'references/I25/evaluations.json')['sets'][label]
            assert scores['raw']['count']==ref['metrics']['metric_missing_count']
            assert scores['effective_mask_seed']==ref['effective_mask_seed']
            assert scores['samples']==ref['samples']
            for other in (suite/'results').glob('*.json'):
                previous=read(other).get('sets',{}).get(label)
                if previous: assert previous['mask_sha256']==scores['mask_sha256'], 'Masks differ across baselines'
            result['sets'][label]=scores;write(dest,result)
            print(name,label,'raw',scores['raw']['mae'],flush=True)
        result['status']='finished';write(dest,result);report(m,suite)
        if name in NEURAL:del model;torch.cuda.empty_cache()
    path=report(m,suite,True);write(suite/'completed.json',dict(time=dt.datetime.now().isoformat(),report=str(path),methods=METHODS))
    write(suite/'state.json',dict(phase='completed',report=str(path)));print('Completed:',path,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--root',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
    suite=a.suite.resolve()
    if a.prepare:prepare(a.root.resolve(),suite);return
    m=setup(suite)
    if a.report_only: print(report(m,suite,(suite/'completed.json').exists()));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Physical GPU1 only: set CUDA_VISIBLE_DEVICES=1')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    lockpath=Path(m['root'])/'outputs/v24-COE/experiments/gpu1.queue.lock'
    with lockpath.open('a') as lock, (suite/'queue.lock').open('a') as own:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(m,suite,a.preflight)
        except BaseException:
            write(suite/'failed.json',dict(time=dt.datetime.now().isoformat(),error=traceback.format_exc()))
            report(m,suite);raise


if __name__=='__main__':main()
