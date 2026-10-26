"""Ten independent trainings: seven input controls and three seed checks."""
import copy
from pathlib import Path

from run_b3_c3 import load
from observation_iteration_config import select_candidate, category_metrics

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'outputs/v25/experiments/taxibj_input_communication/f84c43ccee2d4261'
PREVIOUS = ROOT/'outputs/v25/experiments/observation_iteration/1d6eb79a66c77e49'
CANDIDATES = [f'I{i}' for i in range(54, 61)]
METHODS = {
    'I54': ('I37', '只将专家原生尺度输入中的缺失补位改为0；路由不变'),
    'I55': ('I37', '只让路由的显式值统计读取可见F均值；删除预测值统计'),
    'I56': ('I54/I55', '同时清除专家输入和路由显式统计中的预测补位'),
    'I57': ('I56', '专家每轮不再重复拼入support/position；保留原始观测与mask'),
    'I58': ('I56', 'M/C只输入扣除可见更细尺度后的剩余预算密度；不减当前预测'),
    'I59': ('I56', '仅专家缺失槽加入有界detach当前预测；上限0.1倍可见观测RMS'),
    'I60': ('I57/I58', '预先指定组合：减少重复辅助输入＋静态粗尺度增量观测'),
    'I61': ('I37', '原I37独立训练seed27，与已有seed7/17形成三种子参考'),
    'I62': ('开发集入选方案', '入选方案独立训练seed17，与原I37的I52同种子比较'),
    'I63': ('开发集入选方案', '同一入选方案独立训练seed27，与I61同种子比较'),
}


def options(method):
    if method == 'I61':
        return None
    spec = dict(enabled=True)
    if method in ('I54', 'I56', 'I57', 'I58', 'I60'):
        spec['expert_fill'] = 'zero'
    if method in ('I55', 'I56', 'I57', 'I58', 'I59', 'I60'):
        spec['router_values'] = 'observed'
    if method in ('I57', 'I60'):
        spec['repeated_metadata'] = False
    if method in ('I58', 'I60'):
        spec['coarse_values'] = 'remaining_budget'
    if method == 'I59':
        spec['expert_fill'] = 'bounded_current'
    return spec


def make_job(method, selected=None):
    definition = selected if method in ('I62', 'I63') else method
    if definition not in CANDIDATES + ['I61']:
        raise ValueError('Invalid frozen candidate definition')
    seed = {'I61': 27, 'I62': 17, 'I63': 27}.get(method, 7)
    job = copy.deepcopy(load(REFERENCE/'jobs/taxibj_I37_seed7.json'))
    job.pop('common_initialization', None)
    cfg = job['config'];cfg['seed'] = seed;cfg['data']['loader_seed'] = seed
    spec = options(definition)
    if spec:
        cfg['model']['coe']['input_refinement'] = spec
    variant = f'taxibj_{method}_seed{seed}'
    cfg['experiment_plan'].update(suite='input_refinement', variant=variant,
        method=METHODS[method][1], reference=METHODS[method][0],
        scope='I37 input-only interventions; original communication/history/projection and losses')
    if selected:
        cfg['experiment_plan']['selected_definition'] = selected
    job.update(variant=variant, method=method, name=method.lower()+'_v25_input_refinement',
               reference=METHODS[method][0], defer_test=True)
    return job


def protocols():
    original = load(REFERENCE/'jobs/taxibj_I37_seed7.json')['config']['experiment_plan']['protocol']
    test = copy.deepcopy(original);test['split'] = 'test'
    development = copy.deepcopy(test);development['split'] = 'val'
    confirmation = copy.deepcopy(test)
    for i, spec in enumerate(development['evaluations'].values(), 1):
        spec['seed'] = 2026102200 + i
    for i, spec in enumerate(confirmation['evaluations'].values(), 11):
        spec['seed'] = 2026102200 + i
    return test, development, confirmation
