#!/usr/bin/env python3
"""Portable v25 queue: explicit templates, immutable snapshots, full-state resume."""
import argparse,copy,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/cloud_gpu')]
from run_b3_c3 import load,write,digest
from unpack_data import sha
METHODS=['A1','A2','B1','B2','I01','I02','I03','I04','I05','I06','I07','I08']+[f'S{i:02}' for i in range(1,10)]
SPECS={'S01':('free',[2,2,2,2],[]),'S02':('CMFF',[4,4,4,4],[]),'S03':('CMFF',[1,2,4,8],[]),
       'S04':('CMFF',[2,2,2,2],[2,3,4]),'S05':('CMFF',[2,2,2,2],[2]),'S06':('CMFF',[2,2,2,2],[3]),
       'S07':('CMFF',[2,2,2,2],[4]),'S08':('CFFF',[2,2,2,2],[]),'S09':('CFFF',[2,2,2,2],[4])}


def make_jobs(dataset,methods,rate,epochs=None,batch=32,seed=7,lr=.001,scheduler='cosine',data_root=ROOT,output_root=None,device='cuda'):
    template=load(ROOT/'configs/cloud_gpu/dataset_templates.json')[dataset]
    manifest=load(ROOT/'datasets/manifest.json')['datasets'][dataset];result={}
    for method in methods:
        c=copy.deepcopy(template);c['seed']=seed;c['device']=device;c['output_dir']=str(output_root or ROOT/'outputs/cloud_gpu')
        c['data'].update(batch_size=batch,loader_seed=seed,input_scales='F' if method.startswith('A') else 'FMC');c['data']['mask']['missing_rate']=rate
        for key in ['train_mask_diversity','eval_mask_diversity']:c['data'][key]['rates']=[rate]
        c['train']['epochs']=epochs or c['train']['epochs'];c['train']['lr_main']=lr
        # Same supported scheduler/checkpoint format; eta_min==lr means constant LR.
        c['train']['scheduler']={'type':'cosine','total_epochs':c['train']['epochs'],'eta_min':lr if scheduler=='constant' else .0003,'reach_min_at_last_epoch':True}
        if scheduler=='cosine' and lr<.0003:raise ValueError('Cosine starting LR must be >= 3e-4')
        coe=c['model']['coe'];coe['initial_observation_scales']='F' if method=='I01' else 'FMC'
        coe.pop('scale_delta_exploration',None);coe.pop('observed_history',None);coe.pop('observation_consistency',None)
        coe['core_validation']['communication']='none' if method in ['A1','B1'] else 'conditional'
        if method in SPECS or method=='I02':
            path,ks,steps=SPECS[method] if method in SPECS else ('free',[2]*4,[2,3,4])
            coe['core_validation']['communication']='conditional' if steps else 'none'
            coe['scale_delta_exploration']={'enabled':True,'path':path,'top_k_per_step':ks,'delta_steps':steps,'same_scale_only':method=='S09','scale_gradient':'selected_branch_st','top1_gradient':'selected_probability_st'}
        if method in ('I03','I08'):coe['observed_history']={'enabled':True,'mode':'adaptive'}
        if method=='I04':
            coe['observation_consistency']={'enabled':True,'mode':'full_target_loss'}
            c['data']['return_multiscale_targets']=True
            c['loss']['lambda_observation_consistency']=0.1
        if method in ('I05','I07','I08'):
            coe['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0}
        if method=='I07':coe['observation_consistency']['weight_mode']='uniform'
        if method=='I06':
            coe['observation_consistency']={'enabled':True,'mode':'observed_loss'}
            c['loss']['lambda_observation_consistency']=0.1
        p=load(ROOT/'configs/cloud_gpu/test_protocol.json');p['rate']=rate
        n=f'{dataset}_{method}_seed{seed}'
        c['experiment_plan']={'suite':'cloud_gpu_v25','variant':n,'method':method,'protocol':p,'selection':'ID validation MAE only','cloud_scheduler':scheduler,'scope':'from-scratch cloud run; original mixed-mask protocol'}
        result[n]={'variant':n,'name':method.lower()+'_cloud_v25','method':method,'dataset':dataset,'config':c,'sources':{k:str((Path(data_root)/v['target']).resolve()) for k,v in manifest['files'].items()}}
    return result


def freeze(args):
    output=(args.output_root or ROOT/'outputs/cloud_gpu').resolve();data_root=args.data_root.resolve()
    js=make_jobs(args.dataset,args.methods,args.missing_rate,args.epochs,args.batch_size,args.seed,args.lr,args.scheduler,data_root,output,args.device)
    manifest=load(ROOT/'datasets/manifest.json')['datasets'][args.dataset];data={}
    for split,row in manifest['files'].items():
        path=data_root/row['target']
        if not path.exists():raise FileNotFoundError('Unpack data first: '+str(path))
        if sha(path)!=row['sha256']:raise RuntimeError('Dataset bytes differ from migration manifest: '+str(path))
        data[str(path)]={'size':path.stat().st_size,'sha256':row['sha256']}
    files=[]
    for name in ['src','scripts','configs']:
        files += [p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md'] and '__pycache__' not in p.parts]
    files += [ROOT/'AGENTS.md',ROOT/'README.md',ROOT/'datasets/manifest.json',ROOT/'cloud_environment.json']
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    payload={'project_root':str(ROOT),'jobs':js,'data':data,'source':source,'device':args.device,'gpu':args.gpu if args.gpu is not None else 0,'order':list(js),'preflight':not args.skip_preflight}
    suite=output/'experiments'/args.dataset/digest(payload)[:16];suite.mkdir(parents=True,exist_ok=True)
    immutable(suite/'plan.json',payload)
    for rel,stamp in source.items():
        dst=suite/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(ROOT/rel,dst)
        if sha(dst)!=stamp:raise RuntimeError('Frozen source changed '+rel)
    return suite


def immutable(path,value):
    if path.exists():
        if load(path)!=value:raise RuntimeError('Frozen evidence changed '+str(path))
    else:write(path,value)


def verify(suite,payload):
    for rel,stamp in payload['source'].items():
        if sha(suite/'source_snapshot'/rel)!=stamp:raise RuntimeError('Snapshot changed '+rel)
    for path,row in payload['data'].items():
        if Path(path).stat().st_size!=row['size'] or sha(path)!=row['sha256']:raise RuntimeError('Data changed '+path)


def initialize(suite,js):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    common=None;audit={}
    for name,j in js.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(j['config']['seed']);model=DualBranchSTImputer.from_config(j['config'])
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};base={k:v for k,v in state.items() if not k.startswith(('main_branch.x_history.','main_branch.x_consistency.'))}
        base_hash=state_hash(base)
        if common is None:common=base_hash
        if base_hash!=common:raise RuntimeError('Shared initialization differs '+name)
        path=suite/'initialization'/f'{name}.pth';full=state_hash(state)
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['sha256']!=full or state_hash(saved['model'])!=full:raise RuntimeError('Initialization changed')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save({'model':state,'sha256':full,'seed':j['config']['seed']},tmp);tmp.replace(path)
        j['common_initialization']=str(path);audit[name]={'common_sha256':base_hash,'full_sha256':full,'total_params':sum(p.numel() for p in model.parameters()),'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    immutable(suite/'initialization_audit.json',audit)


def child(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as f:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            while b:=os.read(p.stdout.fileno(),4096):f.write(b);f.flush();sys.stdout.buffer.write(b);sys.stdout.buffer.flush()
            code=p.wait()
        except BaseException:p.terminate();p.wait();raise
    if code:raise RuntimeError(f'Child failed exit {code}; {log}')


def worker(suite,payload):
    from cloud_report import export_report
    import torch,platform
    write(suite/'runtime_environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_name':torch.cuda.get_device_name(0) if payload['device']=='cuda' else None,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time()})
    js=copy.deepcopy(payload['jobs']);initialize(suite,js);immutable(suite/'resolved_jobs.json',js)
    for n,j in js.items():immutable(suite/'jobs'/f'{n}.json',j);immutable(suite/'configs'/f'{n}.json',j['config'])
    acceptance=suite/'acceptance/preflight.json'
    if payload['preflight'] and (not acceptance.exists() or load(acceptance).get('status')!='passed'):
        write(suite/'state.json',{'phase':'preflight','time':time.time()})
        child([sys.executable,'-u',str(suite/'source_snapshot/scripts/cloud_gpu/cloud_preflight.py'),'--suite',str(suite)],suite/'launcher_logs/preflight.log')
    for n,j in js.items():
        run=suite/'runs'/n;receipt=suite/'results'/f'{n}.json';evaluation=suite/'evaluations'/f'{n}.json';p=j['config']['experiment_plan']['protocol']
        immutable(suite/'runs'/f'{n}.json',{'run_dir':str(run)})
        try:
            write(suite/'state.json',{'phase':'training','variant':n,'time':time.time()})
            if receipt.exists():
                r=load(receipt)
                if r.get('status')!='finished' or r['config_sha256']!=digest(j['config']) or r['completed_epochs']!=j['config']['train']['epochs']:raise RuntimeError('Invalid completion receipt')
                if not all((run/'checkpoints'/f'{x}.pth').exists() for x in ['best','last']):raise RuntimeError('Missing completed checkpoint')
            else:child([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/train_four_direction.py'),'--job',str(suite/'jobs'/f'{n}.json'),'--run-dir',str(run),'--result',str(receipt),'--device',payload['device']],suite/'launcher_logs'/f'{n}.train.log')
            write(suite/'state.json',{'phase':'evaluating','variant':n,'time':time.time()})
            e=load(evaluation) if evaluation.exists() else {}
            if e.get('status')!='finished' or e.get('protocol_sha256')!=digest(p) or e.get('config_sha256')!=digest(j['config']) or set(e.get('sets',{}))!=set(p['evaluations']):
                protocol_file=suite/'protocols'/f'{n}.json';immutable(protocol_file,p)
                child([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/evaluate_four_direction.py'),'--checkpoint',str(run/'checkpoints/best.pth'),'--protocol',str(protocol_file),'--npz',j['sources']['test'],'--output',str(evaluation),'--expected-sha',digest(j['config']),'--device',payload['device']],suite/'launcher_logs'/f'{n}.eval.log')
            export_report(suite,js)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'error':repr(error),'time':time.time()});export_report(suite,js);raise
    report=export_report(suite,js)
    if not load(suite/'report_path.json')['complete']:raise RuntimeError('Final evidence audit failed')
    write(suite/'completed.json',{'count':len(js),'time':time.time(),'report':str(report)});write(suite/'state.json',{'phase':'completed','time':time.time()});print('Completed: '+str(report),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj','bikenyc','chap'],default='taxibj');p.add_argument('--methods',nargs='+',choices=METHODS,default=['B1','B2'])
    p.add_argument('--missing-rate',type=float,default=.4);p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=7);p.add_argument('--lr',type=float,default=.001);p.add_argument('--scheduler',choices=['cosine','constant'],default='cosine')
    p.add_argument('--gpu',type=int);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--output-root',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--skip-preflight',action='store_true');p.add_argument('--worker',action='store_true');return p


def main():
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    p=parser();a=p.parse_args()
    if a.resume:
        allowed={'--resume','--gpu','--prepare-only','--worker'}
        if any(t.startswith('--') and t.split('=')[0] not in allowed for t in sys.argv[1:]):p.error('--resume uses frozen settings; only --gpu and launch controls may be changed')
    if not 0<a.missing_rate<1 or a.batch_size<1 or (a.epochs is not None and a.epochs<1) or a.seed<0 or a.lr<=0 or len(a.methods)!=len(set(a.methods)):p.error('Invalid rate, budget, seed, LR or duplicate methods')
    suite=a.resume.resolve() if a.resume else freeze(a);payload=load(suite/'plan.json');verify(suite,payload)
    if a.prepare_only:print(suite);return
    print('Suite: '+str(suite),flush=True)
    if a.worker:worker(suite,payload);return
    gpu=a.gpu if a.gpu is not None else payload['gpu']
    if gpu<0:p.error('GPU index must be nonnegative')
    locks=Path(payload['project_root'])/'outputs/cloud_gpu/locks';locks.mkdir(parents=True,exist_ok=True)
    with (suite/'queue.lock').open('a') as q,(locks/f'{payload["device"]}_{gpu}.lock').open('a') as g:
        fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if payload['device']=='cuda' else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1')
        env.setdefault('MPLCONFIGDIR','/tmp/v25_cloud_mpl')
        write(suite/'runtime_launch.json',{'gpu':gpu,'device':payload['device'],'python':sys.executable,'time':time.time()})
        try:subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/cloud_gpu/run.py'),'--resume',str(suite),'--worker'],env=env,check=True)
        except BaseException as error:write(suite/'state.json',{'phase':'failed','error':repr(error),'time':time.time()});raise

if __name__=='__main__':main()
