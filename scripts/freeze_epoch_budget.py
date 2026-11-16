#!/usr/bin/env python3
"""Freeze the unchanged I37 code and eleven independently trained schedules."""
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from release_utils import load, write, sha256
from epoch_budget_common import ROOT, BASE_COMMIT, INITIAL_HASH, validate_configs


def main():
    p = argparse.ArgumentParser(); p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    suite = a.suite.resolve(); project = a.project.resolve(); source = suite / 'source'
    if source.exists() or (suite / 'plan.json').exists():
        raise FileExistsError('Already frozen; use the frozen queue to resume')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit the implementation before freezing')
    jobs = validate_configs()
    if not (suite / 'cpu_tests.log').read_text().rstrip().endswith('OK'):
        raise RuntimeError('CPU tests have not passed')
    if load(suite / 'preflight_gpu/preflight.json')['status'] != 'passed':
        raise RuntimeError('GPU preflight has not passed')
    # All model and original training/evaluation source stays byte-for-byte best.
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT,
                                     'src', 'scripts', 'configs/test_protocol.json'], cwd=ROOT, text=True).splitlines()
    for name in names:
        expected = subprocess.check_output(['git', 'show', BASE_COMMIT + ':' + name], cwd=ROOT)
        if (ROOT / name).read_bytes() != expected:
            raise RuntimeError('Original implementation changed: ' + name)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source.mkdir()
    for name in subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines():
        src = ROOT / name; dst = source / name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    old = project / 'outputs/v25/experiments/i37_performance/d920258a_seed7_20261026/run'
    reference = suite / 'reference_I37'; reference.mkdir()
    for name in ('config.json', 'result.json', 'evaluation.json', 'training_metadata.json',
                 'initialization.json', 'run_provenance.json', 'runtime.json', 'logs/metrics.jsonl'):
        dst = reference / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(old / name, dst)
    data = project / 'data/TaxiBJ/v25_observed_multiscale/73fe84df49ca05de'
    hashes = load(reference / 'run_provenance.json')['data_sha256']
    for split in ('train', 'val', 'test'):
        if sha256(data / f'taxibj_{split}.npz') != hashes[split]:
            raise RuntimeError('Data differs from I37')
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    torch.set_num_threads(2); torch.manual_seed(7)
    model = DualBranchSTImputer.from_config(load(ROOT / 'configs/i37_performance.json'))
    if state_hash(model.state_dict()) != INITIAL_HASH:
        raise RuntimeError('Initialization changed')
    initial = suite / 'common_initialization.pth'
    torch.save(dict(model=model.state_dict(), sha256=INITIAL_HASH, seed=7), initial)
    plan = dict(version='best_epoch_budget', base_commit=BASE_COMMIT, source_commit=commit,
                source=str(source), python=sys.executable, data_dir=str(data), reference=str(reference),
                report_dir=str(project / 'experments_report'), physical_gpu=1,
                initial_state_sha256=INITIAL_HASH, jobs=jobs,
                selection='ID validation only; six frozen test protocols; single seed exploration')
    write(suite / 'plan.json', plan)
    files = [f for directory in (source, reference) for f in directory.rglob('*') if f.is_file()]
    files += [initial, suite / 'plan.json', suite / 'cpu_tests.log', suite / 'preflight_gpu/preflight.json']
    files += [data / f'taxibj_{s}.npz' for s in ('train', 'val', 'test')]
    write(suite / 'manifest.json', dict(files={str(f): sha256(f) for f in files}))
    cmd = shlex.join([sys.executable, '-u', str(source / 'scripts/queue_epoch_budget.py'), '--suite', str(suite)])
    shell = ('#!/usr/bin/env bash\nset -uo pipefail\n'
             'export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
             + cmd + ' 2>&1 | tee -a ' + shlex.quote(str(suite / 'queue.log')) + '\n'
             'code=${PIPESTATUS[0]}\nprintf "%s\\n" "$code" > ' + shlex.quote(str(suite / 'exit_code')) + '\nexit "$code"\n')
    (suite / 'start_queue.sh').write_text(shell)
    print(dict(commit=commit, jobs=len(jobs), source=str(source)))


if __name__ == '__main__': main()
