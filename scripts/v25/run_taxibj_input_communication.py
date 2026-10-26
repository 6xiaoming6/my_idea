#!/usr/bin/env python3
"""Selectable TaxiBJ I01-I27 jobs; frozen previous suites stay immutable."""
import argparse,copy,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from prepare_data import sha
from run_first_validation import verify
from run_history_plane_interaction import prepare_initialization
import run_four_direction_exploration as base
REFERENCE=ROOT/'outputs/v25/experiments/first_validation/659d3ba9678e1dca'
METHODS={'I01':'H0仅F；CMFF各轮注入对应尺度观测；完整受控差分',
         'I02':'H0三尺度；四轮自由硬选择尺度、共享Top2；完整受控差分',
         'I03':'H0三尺度；CMFF共享Top2；完整受控差分＋自适应预测历史池',
         'I04':'H0三尺度；CMFF共享Top2；完整受控差分＋最终F聚合对完整M/C真值的L1损失',
         'I05':'H0三尺度；CMFF共享Top2；完整受控差分＋可见观测剩余区域自适应非负一致性校正',
         'I06':'H0三尺度；CMFF共享Top2；完整受控差分＋仅可见M/C观测的一致性软损失',
         'I07':'H0三尺度；CMFF共享Top2；完整受控差分＋固定等权剩余区域非负投影',
         'I08':'H0三尺度；CMFF共享Top2；完整受控差分＋自适应预测历史池＋自适应剩余区域投影',
         'I09':'H0三尺度；自由硬选择尺度共享Top2；完整受控差分＋自适应预测历史池＋学习分配投影',
         'I10':'I09基础；前三轮自由硬选择尺度，第四轮固定F；共享Top2＋完整差分＋历史池＋学习分配投影',
         'I11':'I10基础；前三轮按样本无放回选择F/M/C各一次，第四轮固定F；其余组合不变',
         'I12':'固定CMFF复核；H0三尺度＋共享Top2＋完整差分＋历史池＋学习投影；与I08同配置同seed重复'}


COMBINATIONS = {
    'I13': ('I08', 'I08＋投影前可见M/C辅助损失λ=0.1'),
    'I14': ('I10', 'I10＋投影前可见M/C辅助损失λ=0.1；前三轮自由、末轮F'),
    'I15': ('I05', 'I05＋投影前可见M/C辅助损失λ=0.1；无历史池'),
    'I16': ('I03', 'I03＋未投影预测可见M/C辅助损失λ=0.1；无投影'),
    'I17': ('I08', 'I13将辅助损失λ降至0.03'),
    'I18': ('I08', 'I13将辅助损失λ提高至0.3'),
    'I19': ('I08', 'I13将学习分配投影改为固定等权投影；移除分配头'),
    'I20': ('I08', 'I13历史注入gain=0.5；读取、池内容及截断不变'),
    'I21': ('I08', 'I13关闭差分；保留顺序状态、历史池、投影；冻结差分门'),
    'I22': ('I08', 'I13额外增加投影前F缺失位置L1λ=0.1'),
    'I23': ('I14', 'I14双重监督β=0.1：0.9最终F＋0.1原始F；保留0.1可见M/C损失'),
    'I24': ('I14', 'I14双重监督β=0.3：0.7最终F＋0.3原始F；保留0.1可见M/C损失'),
    'I25': ('I24', 'I24的F前后损失权重可学习：β=sigmoid(a)，初始0.3/0.7，始终和为1'),
    'I26': ('I25', 'I25去掉最终投影；保留前三轮自由末轮F、差分、历史、可见M/C损失；F损失合并为单项'),
    'I27': ('I25', 'I25将四轮共享专家池改为四套独立专家池；每轮八选二，其他结构和损失不变'),
    'I29': ('I25', 'I25全程仅F观测；保留自由尺度末轮F、共享Top2、差分与历史；移除M/C投影及相关损失'),
}
EXPLORATION = {
    'I30': ('I25', 'I25只去掉历史池；自由尺度、差分、学习投影和损失不变'),
    'I31': ('I25', 'I25只去掉差分；历史池正常读取，保留顺序状态传递'),
    'I32': ('I25', 'I25只将尺度路由改为固定CMFF；其余不变'),
    'I33': ('I25', 'I25同时去掉历史池和差分；补齐自由尺度通信2×2对照'),
    'I34': ('I25', '固定投影前后F损失权重0.5/0.5，移除可学习混合标量'),
    'I35': ('I25', '固定投影前后F损失权重0.8/0.2，移除可学习混合标量'),
    'I36': ('I25', 'I25增加投影前可见F重建L1，系数0.1，不增加隐藏标签'),
    'I37': ('I34', 'I34增加投影前可见F重建L1，系数0.1'),
    'I38': ('I32', '固定CMFF基础只去历史池；对照I30检验尺度与历史交互'),
    'I39': ('I32', '固定CMFF基础只去差分；对照I31检验尺度与差分交互'),
    'I40': ('I32', '固定CMFF基础同时去历史池和差分；对照I33'),
    'I41': ('I25', 'I25历史读取注入gain=0.5，保留池内容、参数和读写顺序'),
}
COMBINATIONS.update(EXPLORATION)
METHODS.update({name: description for name, (_, description) in COMBINATIONS.items()})


def jobs(reference=REFERENCE, methods=None):
    selected=[f'I{i:02}' for i in range(1,9)] if methods is None else list(methods)
    if not selected or len(set(selected))!=len(selected) or any(m not in METHODS for m in selected):
        raise ValueError('Select distinct methods from registered METHODS')
    origin=load(reference/'jobs/taxibj_B2_seed7.json');result={}
    for method in selected:
        description=METHODS[method]
        if method in COMBINATIONS:
            parent, _ = COMBINATIONS[method]
            j = copy.deepcopy(next(iter(jobs(reference, [parent]).values())))
            c = j['config']; coe = c['model']['coe']; name = f'taxibj_{method}_seed7'
            c['loss']['visible_consistency_aux'] = {
                'weight': {'I17': 0.03, 'I18': 0.3}.get(method, 0.1),
                'source': 'prediction' if method == 'I16' else 'pre_projection'}
            if method == 'I19': coe['observation_consistency']['weight_mode'] = 'uniform'
            if method == 'I20': coe['observed_history']['gain'] = 0.5
            if method == 'I21': coe['core_validation']['communication'] = 'none'
            if method == 'I22': c['loss']['lambda_raw_f'] = 0.1
            if method in ('I23', 'I24'):
                beta = 0.1 if method == 'I23' else 0.3
                c['loss'].update(lambda_final_f=1.0-beta, lambda_raw_f=beta)
            if method == 'I25':
                c['loss'].pop('lambda_final_f');c['loss'].pop('lambda_raw_f')
                c['loss']['learnable_f_mix']={'enabled': True, 'initial_raw_weight': 0.3}
            if method == 'I26':
                coe.pop('observation_consistency')
                c['loss'].pop('learnable_f_mix')
                c['loss']['visible_consistency_aux']['source']='prediction'
            if method == 'I27': coe['expert_sharing']='per_step'
            if method in EXPLORATION:
                if method in ('I30','I33','I38','I40'):
                    coe.pop('observed_history')
                if method in ('I31','I33','I39','I40'):
                    coe['core_validation']['communication']='none'
                    if 'scale_delta_exploration' in coe:
                        coe['scale_delta_exploration']['delta_steps']=[]
                if method == 'I32': coe.pop('scale_delta_exploration')
                if method in ('I34','I35'):
                    c['loss'].pop('learnable_f_mix')
                    beta = 0.5 if method == 'I34' else 0.8
                    c['loss'].update(lambda_raw_f=beta, lambda_final_f=1.0-beta)
                if method in ('I36','I37'): c['loss']['lambda_visible_raw_f']=0.1
                if method == 'I41': coe['observed_history']['gain']=0.5
            if method == 'I29':
                c['data']['input_scales']='F'
                coe['initial_observation_scales']='F'
                coe.pop('observation_consistency')
                c['loss'].pop('learnable_f_mix')
                c['loss'].pop('visible_consistency_aux')
                c['data']['return_multiscale_targets']=False
            c['experiment_plan'].update(variant=name, method=description,
                reference=parent if method in ('I13','I14','I15','I16','I23','I24','I25','I26','I27','I29') else 'I13',
                scope='I13-I22 combination exploration; observed-only auxiliaries; single seed; ID threshold 8.8709')
            if method in ('I23','I24'):
                c['experiment_plan']['scope']='I14 normalized pre/post projection F supervision; common initialization; insert after I18 before I19; same six protocols'
            if method=='I25':
                c['experiment_plan']['scope']='learnable convex F loss weights from 0.3 raw/0.7 final; replaces user-paused I24; common initialization, no checkpoint warm start'
            if method=='I26':
                c['experiment_plan']['scope']='I25 minus output projection; identical F losses collapse to one and remove redundant beta; observed-only M/C auxiliary; common initialization; priority before resuming I19-I22'
            if method=='I27':
                c['experiment_plan']['scope']='I25 independent per-round expert pools cloned from common shared initialization; native Top2, original learned projection and beta retained; after I26 before I19-I22'
            if method=='I29':
                c['experiment_plan']['scope']='Strict F-only observations for baseline comparison; internal feature scales retained; no M/C supervision/projection; no warm start; same F masks and six protocols'
            if method in EXPLORATION:
                c['experiment_plan'].update(reference=parent,
                    scope='I30-I41 I25 robustness exploration; 2x2x2 scale/history/delta controls and raw reconstruction supervision; independent common initialization; 100 epochs, same six protocols')
            j.update(variant=name, method=method, name=method.lower()+'_v25_taxibj_input_communication',
                     reference=c['experiment_plan']['reference'])
            result[name] = j
            continue
        j=copy.deepcopy(origin);j.pop('common_initialization',None);c=j['config'];name=f'taxibj_{method}_seed7'
        c['model']['coe']['initial_observation_scales']='F' if method=='I01' else 'FMC'
        if method in ('I02','I09','I10','I11'):c['model']['coe']['scale_delta_exploration']={'enabled':True,'path':'free','top_k_per_step':[2]*4,'delta_steps':[2,3,4],'same_scale_only':False,'scale_gradient':'selected_branch_st','top1_gradient':'selected_probability_st'}
        if method in ('I10','I11'):c['model']['coe']['scale_delta_exploration']['last_round_fine']=True
        if method=='I11':c['model']['coe']['scale_delta_exploration']['first_three_unique']=True
        if method=='I03':c['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
        if method=='I04':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'full_target_loss'}
            c['loss']['lambda_observation_consistency']=0.1
            c['data']['return_multiscale_targets']=True
        if method=='I05':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0}
        if method=='I06':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'observed_loss'}
            c['loss']['lambda_observation_consistency']=0.1
        if method=='I07':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0,'weight_mode':'uniform'}
        if method in ('I08','I09','I10','I11','I12'):
            c['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0}
        c['experiment_plan'].update(suite='v25_taxibj_input_communication',variant=name,method=description,reference='B2',scope='three requested interventions; 100 epochs original B2 protocol')
        if method in ('I04','I05'):
            c['experiment_plan']['scope']='consistency followup; full-target soft loss versus observed-only adaptive residual projection; no history; 100 epochs original B2 protocol'
        if method in ('I06','I07','I08'):
            c['experiment_plan']['scope']='eight-run mechanism controls: observed-only soft loss, fixed projection, history plus learned projection; original B2 protocol'
        if method=='I09':
            c['experiment_plan']['scope']='single requested full combination; I08 plus I02 free-scale training; original B2 100-epoch protocol'
        if method=='I10':
            c['experiment_plan']['scope']='I09 control: free first three scales, hard final F; train from common initialization for 100 epochs; I09 paused at user request'
        if method=='I11':
            c['experiment_plan']['scope']='I10 control: per-sample scale selection without replacement for rounds 1-3, final F; independent 100 epochs after I10 training/evaluation/report complete'
        if method=='I12':
            c['experiment_plan']['scope']='requested fixed CMFF recheck; same computational config and seed as I08, not a new architecture or independent training-seed replicate; prepare only until explicitly launched'
            c['experiment_plan']['replicates_method']='I08'
        j.update(variant=name,method=method,name=method.lower()+'_v25_taxibj_input_communication',reference='B2')
        result[name]=j
    return result


def freeze(methods=None):
    js=jobs(methods=methods);old=load(REFERENCE/'plan.json')
    # Check historical data before inheriting a reference comparison.
    for path,stamp in old['data'].items():
        if sha(path)!=stamp['sha256']:raise RuntimeError('B2 dataset changed '+path)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').glob('*.json'))
    files += [ROOT/'AGENTS.md',ROOT/'scripts/v25/README_TAXIBJ_INPUT_COMMUNICATION.md',ROOT/'tests/test_v25_taxibj_input_communication.py',ROOT/'tests/test_v25_observed_multiscale.py',ROOT/'tests/test_v25_observation_consistency.py',ROOT/'tests/test_v25_consistency_controls.py',ROOT/'tests/test_v25_full_combination.py',ROOT/'tests/test_v25_permutation_scales.py',ROOT/'tests/test_v25_queue_dependency.py']
    files += [ROOT/'tests/test_v25_combination_aux.py', ROOT/'scripts/v25/README_COMBINATION_EXPLORATION.md']
    files += [ROOT/'tests/test_v25_dual_supervision.py']
    files += [ROOT/'tests/test_v25_learnable_loss_mix.py']
    files += [ROOT/'tests/test_v25_no_projection.py']
    files += [ROOT/'tests/test_v25_independent_experts.py',ROOT/'tests/test_v25_f_only.py',ROOT/'scripts/v25/README_F_ONLY.md']
    if any(j['method'] in EXPLORATION for j in js.values()):
        files += [ROOT/'tests/test_v25_robustness_exploration.py',ROOT/'scripts/v25/README_ROBUSTNESS_EXPLORATION.md']
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))};refs={}
    for method in ['A2','B1','B2']:
        n=f'taxibj_{method}_seed7';r=load(REFERENCE/'results'/f'{n}.json');run=Path(r['run_dir'])
        for label,p in [('receipt',REFERENCE/'results'/f'{n}.json'),('evaluation',REFERENCE/'evaluations'/f'{n}.json'),('job',REFERENCE/'jobs'/f'{n}.json'),('config',run/'config.json'),('metadata',run/'training_metadata.json'),('history',run/'logs/metrics.jsonl')]:refs[f'{method}/{label}.json']={'path':str(p),'sha256':sha(p)}
    init=REFERENCE/'initialization/taxibj_B2_seed7.pth';refs['initialization.pth']={'path':str(init),'sha256':sha(init)}
    if any(j['method'] in ('I10','I11') for j in js.values()):
        paused=ROOT/'outputs/v25/experiments/taxibj_input_communication/70c7693b8f5709d1'
        record=load(paused/'paused_evaluation.json');evaluation=load(paused/'evaluations/taxibj_I09_seed7.json')
        run=Path(load(paused/'runs/taxibj_I09_seed7.json')['run_dir'])
        if record['status']!='paused_by_user' or evaluation['status']!='finished' or sha(record['checkpoint'])!=record['checkpoint_sha256']:
            raise RuntimeError('Paused I09 best evaluation must be complete and unchanged')
        for label,p in [('evaluation',paused/'evaluations/taxibj_I09_seed7.json'),('paused',paused/'paused_evaluation.json'),('config',run/'config.json'),('metadata',run/'training_metadata.json'),('history',run/'logs/metrics.jsonl')]:
            refs[f'I09/{label}.json']={'path':str(p),'sha256':sha(p)}
    payload={'project_root':str(ROOT),'jobs':js,'order':list(js),'gpu':1,'source':sources,'data':old['data'],'references':refs,'reference_suite':str(REFERENCE),'origin':{'branch':subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()}}
    if any(j['method'] in ('I06','I07','I08','I09','I10','I11','I12') for j in js.values()):
        # These already frozen suites are read-only comparison dependencies.
        # Never import their jobs into this training order or retrain them here.
        previous=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        payload['comparison_suites']={str(previous/name):sha(previous/name/'plan.json')
                                      for name in ('ea1ca776ade869d6','b5925acd4615e143')}
        if any(j['method'] in ('I09','I10','I11','I12') for j in js.values()):
            payload['comparison_suites'][str(previous/'0670f12cc0d77181')]=sha(previous/'0670f12cc0d77181/plan.json')
        if any(j['method']=='I11' for j in js.values()):
            dependency=previous/'a8c5e28f7e93b169'
            payload['run_after_suite']={'path':str(dependency),'plan_sha256':sha(dependency/'plan.json')}
            payload['comparison_suites'][str(dependency)]=sha(dependency/'plan.json')
        if any(j['method']=='I12' for j in js.values()):
            # Read-only comparisons; never schedule I08/I10/I11 or a waiter.
            for name in ('a8c5e28f7e93b169','8cc101c9c2074582'):
                payload['comparison_suites'][str(previous/name)]=sha(previous/name/'plan.json')
    if any(j['method'] in COMBINATIONS for j in js.values()):
        previous = ROOT/'outputs/v25/experiments/taxibj_input_communication'
        payload['comparison_suites'] = {str(previous/name): sha(previous/name/'plan.json')
            for name in ('ea1ca776ade869d6','b5925acd4615e143','0670f12cc0d77181',
                         'a8c5e28f7e93b169','8cc101c9c2074582','cc54874eb82b0d91')}
        payload['selection'] = {'id_mae_max': 8.8709, 'reference': 'I08',
            'ood_categories': [['unseen_combinations','unseen_combinations_repeat'],
                               ['unseen_geometry','unseen_geometry_repeat'], ['unseen_triple']],
            'ranking': ['category_macro_relative_mae','category_macro_relative_rmse','id_mae'],
            'training_seeds': [7]}
    if any(j['method'] in ('I23','I24','I25') for j in js.values()):
        dependency = ROOT/'outputs/v25/experiments/taxibj_input_communication/22d4ae99d3e86986'
        if set(js) & set(load(dependency/'plan.json')['jobs']):
            raise ValueError('Do not schedule dependency jobs alongside their followup')
        payload['interleave_after'] = {'path': str(dependency), 'plan_sha256': sha(dependency/'plan.json'), 'job': 'taxibj_I18_seed7'}
        payload['comparison_methods'] = {str(dependency): [f'I{i:02}' for i in range(13,19)]}
        payload['comparison_suites'][str(dependency)] = sha(dependency/'plan.json')
    if any(j['method']=='I25' for j in js.values()):
        paused=ROOT/'outputs/v25/experiments/taxibj_input_communication/37ae81a02eea687f'
        if load(paused/'paused.json').get('status')!='paused_by_user':raise RuntimeError('I24 must be explicitly paused')
        payload['adopt_suspended_from']={'path':str(paused), 'pause_sha256':sha(paused/'paused.json'),
                                       'lease_sha256':sha(paused/'interleave_lease.json')}
        payload['comparison_suites'][str(paused)]=sha(paused/'plan.json')
        payload['comparison_methods'][str(paused)]=['I23']
        for label,p in [('paused',paused/'paused.json'),('job',paused/'jobs/taxibj_I24_seed7.json')]:
            refs[f'I24/{label}.json']={'path':str(p),'sha256':sha(p)}
    if any(j['method']=='I26' for j in js.values()):
        previous=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        old_queue=previous/'22d4ae99d3e86986'
        pause=old_queue/'paused_for_i26.json'
        if load(pause).get('status')!='paused_by_user' or load(old_queue/'state.json').get('phase')!='paused_by_user':
            raise RuntimeError('I19 must be paused before freezing the priority I26 queue')
        payload['resume_after_priority']={'path':str(old_queue),'plan_sha256':sha(old_queue/'plan.json'),
            'pause_path':str(pause),'pause_sha256':sha(pause)}
        payload['comparison_methods']={str(old_queue):[f'I{i:02}' for i in range(13,19)],
            str(previous/'37ae81a02eea687f'):['I23']}
        for name in ('22d4ae99d3e86986','37ae81a02eea687f','2c07a1e3f7f177b7'):
            payload['comparison_suites'][str(previous/name)]=sha(previous/name/'plan.json')
        for label,p in [('I19/paused.json',pause),('I24/paused.json',previous/'37ae81a02eea687f/paused.json')]:
            refs[label]={'path':str(p),'sha256':sha(p)}
    if any(j['method']=='I27' for j in js.values()):
        previous=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        dependency=previous/'8f67d8cc2cbf072e'
        hold=dependency/'hold_resume_for_i27.json'
        payload['insert_before_resume']={'path':str(dependency),'plan_sha256':sha(dependency/'plan.json'),
            'controller_hold':str(hold),'controller_hold_sha256':sha(hold)}
        payload['comparison_methods']={str(previous/'22d4ae99d3e86986'):[f'I{i:02}' for i in range(13,19)],
            str(previous/'37ae81a02eea687f'):['I23']}
        for name in ('22d4ae99d3e86986','37ae81a02eea687f','2c07a1e3f7f177b7','8f67d8cc2cbf072e'):
            payload['comparison_suites'][str(previous/name)]=sha(previous/name/'plan.json')
        for label,p in [('I26/controller_hold.json',hold),('I24/paused.json',previous/'37ae81a02eea687f/paused.json'),
                        ('I19/paused.json',previous/'22d4ae99d3e86986/paused_for_i26.json')]:
            refs[label]={'path':str(p),'sha256':sha(p)}
    if any(j['method']=='I29' for j in js.values()):
        if len(js)!=1:raise ValueError('I29 must run as an isolated F-only suite')
        payload.pop('selection',None)
        payload.pop('comparison_suites',None)
        parent=ROOT/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7'
        for kind in ('jobs','results','evaluations','configs'):
            path=parent/kind/'taxibj_I25_seed7.json'
            refs[f'I25/{kind}.json']={'path':str(path),'sha256':sha(path)}
        path=parent/'initialization/taxibj_I25_seed7.pth'
        refs['I25/initialization.pth']={'path':str(path),'sha256':sha(path)}
        baseline=ROOT/'outputs/v25/experiments/simple_baselines/20261015_161454'
        for method in ('ChannelMean','NodeMean','TemporalLinear','SpatialMean','SoftImpute'):
            path=baseline/'results'/f'{method}.json'
            refs[f'baselines/{method}.json']={'path':str(path),'sha256':sha(path)}
    if any(j['method'] in EXPLORATION for j in js.values()):
        if not all(j['method'] in EXPLORATION for j in js.values()):
            raise ValueError('Keep I30-I41 isolated from older queues')
        previous=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        parent=previous/'2c07a1e3f7f177b7'
        payload['comparison_suites'][str(parent)]=sha(parent/'plan.json')
        payload['selection']['id_mae_max']=8.8709
        payload['selection']['secondary_id_i25_10pct']=load(parent/'evaluations/taxibj_I25_seed7.json')['sets']['in_distribution']['metrics']['mae']*1.1
        payload['exploration_reference']='I25'
        for kind in ('jobs','results','evaluations','configs'):
            path=parent/kind/'taxibj_I25_seed7.json'
            refs[f'I25/{kind}.json']={'path':str(path),'sha256':sha(path)}
        path=parent/'initialization/taxibj_I25_seed7.pth'
        refs['I25/initialization.pth']={'path':str(path),'sha256':sha(path)}
    suite=ROOT/'outputs/v25/experiments/taxibj_input_communication'/digest(payload)[:16];suite.mkdir(parents=True,exist_ok=True);base.frozen(suite/'plan.json',payload)
    for name,stamp in sources.items():
        target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(ROOT/name,target)
        if sha(target)!=stamp:raise RuntimeError('Frozen source mismatch '+name)
    for name,row in refs.items():
        target=suite/'references'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(row['path'],target)
        if sha(target)!=row['sha256']:raise RuntimeError('Frozen reference mismatch '+name)
    return suite


def gpu1_only(gpus=(1,)):
    if list(gpus)!=[1]:raise ValueError('Only physical GPU1 is authorized')


def worker(suite,payload):
    from report_taxibj_input_communication import export_report
    base.check_gpu_idle=gpu1_only
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js,suite/'references/initialization.pth')
    if any(j['method'] in EXPLORATION for j in js.values()):
        import torch
        original=torch.load(suite/'references/I25/initialization.pth',map_location='cpu',weights_only=False)['model']
        audit={}
        for name,j in js.items():
            current=torch.load(j['common_initialization'],map_location='cpu',weights_only=False)['model']
            if any(k not in original or not torch.equal(v,original[k]) for k,v in current.items()):
                raise RuntimeError('Exploration retained parameter differs from I25 initialization: '+name)
            removed=sorted(set(original)-set(current))
            if any(not (k.startswith('main_branch.x_history.') or k=='main_branch.x_loss_mix_logit') for k in removed):
                raise RuntimeError('Unexpected removed parameters: '+name)
            audit[name]={'status':'passed','retained_tensors':len(current),'removed_keys':removed}
        write(suite/'i25_initialization_parity.json',audit)
    if any(j['method']=='I29' for j in js.values()):
        import torch
        original=torch.load(suite/'references/I25/initialization.pth',map_location='cpu',weights_only=False)['model']
        current=torch.load(js['taxibj_I29_seed7']['common_initialization'],map_location='cpu',weights_only=False)['model']
        removed=set(original)-set(current)
        if any(not (k.startswith('main_branch.x_consistency.') or k=='main_branch.x_loss_mix_logit') for k in removed):
            raise RuntimeError('Unexpected removed I25 parameters')
        if any(k not in original or not torch.equal(v,original[k]) for k,v in current.items()):
            raise RuntimeError('I29 common initialization differs from I25')
        write(suite/'i25_initialization_parity.json',{'status':'passed','shared_tensors':len(current),'removed_keys':sorted(removed)})
    base.frozen(suite/'resolved_jobs.json',js)
    for n,j in js.items():base.frozen(suite/'jobs'/f'{n}.json',j);base.frozen(suite/'configs'/f'{n}.json',j['config'])
    if not (suite/'started.json').exists():write(suite/'started.json',{'gpu':1,'time':time.time(),'order':list(js)})
    acceptance=suite/'acceptance/preflight.json'
    if not acceptance.exists() or load(acceptance).get('status')!='passed':
        write(suite/'state.json',{'phase':'preflight','gpu':1,'time':time.time()})
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight_taxibj_input_communication.py'),'--suite',str(suite)],suite/'launcher_logs/preflight.log',1,visible=True)
    for n,j in js.items():
        try:
            for path,stamp in payload['data'].items():
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Data changed '+path)
            write(suite/'state.json',{'phase':'training','variant':n,'gpu':1,'time':time.time()})
            base.launch(suite,j,1);r=base.receipt(suite,j)
            write(suite/'state.json',{'phase':'evaluating','variant':n,'gpu':1,'time':time.time()})
            base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],j['config']['experiment_plan']['protocol'],j['sources']['test'],'evaluations',1)
            export_report(Path(payload['project_root']),suite,js)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'time':time.time(),'error':repr(error)})
            export_report(Path(payload['project_root']),suite,js);raise
    report=export_report(Path(payload['project_root']),suite,js)
    if not load(suite/'report_path.json')['complete']:raise RuntimeError('Final report audit failed')
    write(suite/'completed.json',{'count':len(js),'time':time.time(),'report':str(report)})
    write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(report)})
    print('Completed '+str(suite),flush=True)


def main():
    # New adaptive-selection suite has its own controller; never use the legacy
    # train-then-test loop before I53 has been selected on validation evidence.
    if '--observation-iteration' in sys.argv:
        sys.argv.remove('--observation-iteration')
        from run_observation_iteration import main as observation_main
        return observation_main()
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--methods',nargs='+',choices=list(METHODS),help='Select new jobs only; I11 must wait for the frozen I10 suite');p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--summary-only',action='store_true');a=p.parse_args()
    if a.suite and a.methods:p.error('--methods cannot modify an existing --suite')
    suite=a.suite.resolve() if a.suite else freeze(a.methods);payload=load(suite/'plan.json');verify(suite,payload)
    for name,row in payload['references'].items():
        if sha(suite/'references'/name)!=row['sha256']:raise RuntimeError('Reference changed '+name)
    if a.prepare_only:print(suite);return
    if a.summary_only:
        from report_taxibj_input_communication import export_report
        print(export_report(Path(payload['project_root']),suite,load(suite/'resolved_jobs.json')));return
    if payload.get('insert_before_resume'):
        from run_before_priority_resume import validate_launch
        validate_launch(payload)
    if payload.get('run_after_suite'):
        from run_after_suite import dependency_status
        status,detail=dependency_status(payload['run_after_suite'])
        if status!='ready':raise RuntimeError('Dependency not ready; use run_after_suite.py to queue: '+str(detail))
    if payload.get('interleave_after'):
        from run_interleaved import validate_launch
        if not a.worker:raise RuntimeError('Use run_interleaved.py for the I18 boundary insertion')
        validate_launch(suite,payload)
    if a.worker:worker(suite,payload);return
    gpu_lock=Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock';gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock,gpu_lock.open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHONFAULTHANDLER='1')
        try:subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_taxibj_input_communication.py'),'--suite',str(suite),'--worker'],env=env,check=True)
        except BaseException as error:
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)});raise

if __name__=='__main__':main()
