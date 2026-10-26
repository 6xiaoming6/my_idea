"""Frozen I42-I53 definitions; no outcome-dependent method construction."""
import copy
from pathlib import Path
from run_b3_c3 import load

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'outputs/v25/experiments/taxibj_input_communication/f84c43ccee2d4261'
METHODS={
 'I42':('I37','固定MMFF；原数据流和通信'),
 'I43':('I37','逐轮刷新补位；专家路由读取所选尺度融合输入'),
 'I44':('I43','去重的剩余观测差异替代绝对观测注入'),
 'I45':('I44','剩余观测差异驱动的尺度与专家路由偏置'),
 'I46':('I44','预测空间总量/细节差分替代隐藏差分'),
 'I47':('I44','去总量、去最近变化的历史创新'),
 'I48':('I44','已选Top2按分数串行传递剩余观测差异'),
 'I49':('I44','观测区域归属限制粗尺度状态更新'),
 'I50':('I45','新差分＋新历史'),
 'I51':('I50','专家残差传递＋区域状态更新'),
 'I52':('I37','原I37独立训练seed17'),
 'I53':('validation selection','冻结候选中开发协议最优者独立训练seed17')}
CANDIDATES=list(METHODS)[:10]

def options(method):
    if method=='I52':return None
    if method=='I42':return dict(enabled=True,fixed_path='MMFF')
    result=dict(enabled=True,refresh_completion=True,expert_router_input='native')
    if method!='I43':result['round_input']='remaining_residual'
    if method in ('I45','I50','I51'):result['routing_evidence']=True
    if method in ('I46','I50','I51'):result['delta']='physical_split'
    if method in ('I47','I50','I51'):result['history']='orthogonal_innovation'
    if method in ('I48','I51'):result['pair']='residual_relay'
    if method in ('I49','I51'):result['state_update']='ownership_correction'
    return result

def make_job(method,seed=None,selected=None,reference=REFERENCE):
    definition=selected if method=='I53' else method
    if definition not in CANDIDATES+['I52']:raise ValueError('I53 requires a frozen candidate')
    seed=(17 if method in ('I52','I53') else 7) if seed is None else seed
    j=copy.deepcopy(load(reference/'jobs/taxibj_I37_seed7.json'))
    j.pop('common_initialization',None)
    cfg=j['config'];cfg['seed']=seed;cfg['data']['loader_seed']=seed
    spec=options(definition)
    if spec:cfg['model']['coe']['observation_iteration']=spec
    name=f'taxibj_{method}_seed{seed}'
    cfg['experiment_plan'].update(suite='observation_iteration',variant=name,method=METHODS[method][1],
        reference=METHODS[method][0],scope='Observed-only iteration; independent initialization; development-only I53 selection')
    if selected:cfg['experiment_plan']['selected_definition']=selected
    j.update(variant=name,method=method,name=method.lower()+'_v25_observation_iteration',reference=METHODS[method][0],defer_test=True)
    return j

def protocols(reference=REFERENCE):
    original=load(reference/'jobs/taxibj_I37_seed7.json')['config']['experiment_plan']['protocol']
    test=copy.deepcopy(original);test['split']='test'
    dev=copy.deepcopy(test);dev['split']='val'
    for i,spec in enumerate(dev['evaluations'].values(),1):spec['seed']=2026101700+i
    confirm=copy.deepcopy(dev);confirm['split']='test'
    return test,dev,confirm

CATEGORIES=(('unseen_combinations','unseen_combinations_repeat'),
            ('unseen_geometry','unseen_geometry_repeat'),('unseen_triple',))

def category_metrics(record,key):
    return [sum(record['sets'][n]['metrics'][key] for n in names)/len(names) for names in CATEGORIES]

def select_candidate(reference,records,parameters):
    if reference['split']!='val' or any(v['split']!='val' for v in records.values()):
        raise ValueError('Method selection cannot use test evidence')
    rows=[];rid=reference['sets']['in_distribution']['metrics']['mae']
    for name,r in records.items():
        mid=r['sets']['in_distribution']['metrics']['mae'];scores={}
        for metric in ('mae','rmse'):
            scores[metric]=sum(a/b for a,b in zip(category_metrics(r,metric),category_metrics(reference,metric)))/3
        rows.append(dict(method=name,id_mae=mid,id_ratio=mid/rid,eligible=mid<=1.03*rid,
                         ood_mae_ratio=scores['mae'],ood_rmse_ratio=scores['rmse'],parameters=parameters[name]))
    eligible=[r for r in rows if r['eligible']]
    rank=lambda r:(r['ood_mae_ratio'],r['ood_rmse_ratio'],r['id_mae'],r['parameters'],r['method'])
    selected=min(eligible,key=rank) if eligible else min(rows,key=lambda r:(r['id_ratio'],*rank(r)))
    return dict(selected=selected['method'],qualified_for_replication=bool(eligible),rows=sorted(rows,key=rank),
                selection_split='val',reference_id_mae=rid,id_limit=rid*1.03)
