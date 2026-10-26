"""I64-I75: twelve predeclared seed7 input controls, without new training seeds."""
import copy
from pathlib import Path
from run_b3_c3 import load
from observation_iteration_config import category_metrics, select_candidate

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT/'model_designs/20261023_v25_I64-I75最后一轮seed7优化方案.json'
REFERENCE = ROOT/'outputs/v25/experiments/taxibj_input_communication/f84c43ccee2d4261'
PREVIOUS = ROOT/'outputs/v25/experiments/input_refinement/ad274d0529acb794'
DEFINITIONS = {s['id']: s for s in load(DESIGN)['experiments']}
CANDIDATES = list(DEFINITIONS)
METHODS = {n: (s['base'], s['change']) for n, s in DEFINITIONS.items()}


def options(method):
    s = DEFINITIONS[method]
    fill = {'original_filled_f': 'initial', 'zero': 'zero',
            'visible_f_plus_scaled_initial_prediction': 'visible_initial'}[s['expert_fallback']]
    # The gain has no meaning in the historical whole-slot-zero controls.
    return dict(enabled=True, component_diagnostics=True, expert_fill=fill, router_values=s['router_values'],
                support_gain=s['support_gain'], position_gain=s['position_gain'],
                prediction_gain=s['prediction_gain'] if fill == 'visible_initial' else 1.)


def make_job(method):
    j = copy.deepcopy(load(REFERENCE/'jobs/taxibj_I37_seed7.json'))
    j.pop('common_initialization', None)
    cfg = j['config']; cfg['model']['coe']['input_refinement'] = options(method)
    name = f'taxibj_{method}_seed7'
    cfg['experiment_plan'].update(suite='final_input', variant=name, method=METHODS[method][1],
        reference=METHODS[method][0], scope='Frozen final seed7 input exploration; original I37 rounds and losses')
    j.update(variant=name, method=method, name=method.lower()+'_v25_final_input',
             reference=METHODS[method][0], defer_test=True)
    return j


def protocols():
    test = copy.deepcopy(load(REFERENCE/'jobs/taxibj_I37_seed7.json')['config']['experiment_plan']['protocol'])
    test['split'] = 'test'
    dev = copy.deepcopy(test); dev['split'] = 'val'
    confirm = copy.deepcopy(test)
    for i, spec in enumerate(dev['evaluations'].values(), 1): spec['seed'] = 2026102300+i
    for i, spec in enumerate(confirm['evaluations'].values(), 11): spec['seed'] = 2026102300+i
    return test, dev, confirm


def choose(reference, records, parameters):
    result = select_candidate(reference, records, parameters)
    # This batch has no fallback replication and must not force a new model.
    if not result['qualified_for_replication']: result['selected'] = None
    result['qualified_by_development_id'] = result.pop('qualified_for_replication')
    result['no_extra_training'] = True
    return result
