"""Contracts for the I37 epoch / minimum learning rate experiment suite."""
from pathlib import Path
import copy
import math
from release_utils import load, digest

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '3ab02340'
INITIAL_HASH = '2baf7361f1aa2308e5f2a55c07fd657e7d5feea87dd296ca071406aec3179f99'
CATEGORIES = {
    '二元组合': ['unseen_combinations', 'unseen_combinations_repeat'],
    '未见形态': ['unseen_geometry', 'unseen_geometry_repeat'],
    '三元组合': ['unseen_triple'],
}


def validate_configs(root=ROOT):
    root = Path(root)
    base = load(root / 'configs/i37_performance.json')
    jobs = load(root / 'configs/epoch_budget/suite.json')
    expected = {(n, e) for n in (100, 120, 140, 160, 180, 200)
                for e in (3e-4, 1e-5)} - {(100, 3e-4)}
    assert len(jobs) == len({j['name'] for j in jobs}) == 11
    assert {(j['epochs'], j['eta_min']) for j in jobs} == expected
    for job in jobs:
        cfg = load(root / job['config'])
        assert cfg['train']['epochs'] == job['epochs']
        assert cfg['train']['scheduler']['total_epochs'] == job['epochs']
        assert cfg['train']['scheduler']['eta_min'] == job['eta_min']
        normalized = copy.deepcopy(cfg)
        normalized['train']['epochs'] = base['train']['epochs']
        normalized['train']['scheduler'] = copy.deepcopy(base['train']['scheduler'])
        assert normalized == base, job['name']
        original_scheduler = dict(cfg['train']['scheduler'])
        original_scheduler.update(total_epochs=100, eta_min=3e-4)
        assert original_scheduler == base['train']['scheduler'], job['name']
    return jobs


def completed(run, job, cfg, protocol):
    """Validate finished receipts; incomplete output may be resumed, not skipped."""
    run = Path(run)
    if not (run / 'result.json').exists() or not (run / 'evaluation.json').exists():
        return False
    result = load(run / 'result.json'); evaluation = load(run / 'evaluation.json')
    if result.get('status') != 'finished' or evaluation.get('status') != 'finished':
        return False
    assert result['completed_epochs'] == job['epochs'], 'Wrong training budget'
    assert result['config_sha256'] == evaluation['config_sha256'] == digest(cfg)
    assert evaluation['protocol_sha256'] == digest(protocol)
    assert evaluation['best_epoch'] == result['best_epoch']
    assert evaluation['selection'] == 'ID validation only'
    assert evaluation['split'] == 'test'
    assert set(evaluation['sets']) == set(protocol['evaluations'])
    for row in evaluation['sets'].values():
        assert all(math.isfinite(row['metrics'][m]) for m in ('mae', 'rmse'))
    assert all((run / 'checkpoints' / f'{k}.pth').is_file() for k in ('best', 'last'))
    assert load(run / 'initialization.json')['full_state_sha256'] == INITIAL_HASH
    return True


def scores(evaluation, reference):
    result = {}
    for metric in ('mae', 'rmse'):
        cats = {name: sum(evaluation['sets'][k]['metrics'][metric] for k in keys) / len(keys)
                for name, keys in CATEGORIES.items()}
        refs = {name: sum(reference['sets'][k]['metrics'][metric] for k in keys) / len(keys)
                for name, keys in CATEGORIES.items()}
        ratios = {k: cats[k] / refs[k] for k in cats}
        result[metric] = dict(categories=cats, mean=sum(cats.values()) / 3,
                              ratio=sum(ratios.values()) / 3,
                              worst_pct=100 * (max(ratios.values()) - 1))
    return result
