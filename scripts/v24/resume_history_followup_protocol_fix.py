#!/usr/bin/env python3
"""Audited evaluation-path repair; execute the original frozen Y model/queue."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time


def scoped_evaluate_record(base, suite, name, cp, sha, protocol, npz, group, gpu):
    output = suite/group/f'{name}.json'; ph = base.digest(protocol)
    if output.exists():
        r = base.load(output)
        if (r.get('status') == 'finished' and r['protocol_sha256'] == ph
                and r['config_sha256'] == sha and r['checkpoint'] == str(Path(cp).resolve())
                and r.get('npz') == str(Path(npz).resolve())):
            return r
    # The protocol identity, rather than the whole suite, defines this frozen file.
    protocolpath = suite/'protocols'/group/f'{ph}.json'
    base.frozen(protocolpath, protocol)
    base.subprocess_run([sys.executable, '-u', str(suite/'source_snapshot/scripts/v24/evaluate_four_direction.py'),
        '--checkpoint', str(cp), '--protocol', str(protocolpath), '--npz', npz,
        '--output', str(output), '--expected-sha', sha], suite/'launcher_logs'/f'{name}.{group}.log', gpu)
    r = base.load(output)
    if r['status'] != 'finished':
        raise RuntimeError('Incomplete evaluation')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--suite', type=Path, required=True); a = p.parse_args()
    suite = a.suite.resolve(); snapshot = suite/'source_snapshot'
    sys.path[:0] = [str(snapshot/'scripts/v24'), str(snapshot/'src')]
    import run_history_followup as runner
    if Path(runner.__file__).resolve() != snapshot/'scripts/v24/run_history_followup.py':
        raise RuntimeError('Must use the original frozen queue')
    payload = runner.load(suite/'plan.json')
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.validate(suite, payload)
        evidence = dict(started=time.time(), repair='protocol path scoped by protocol digest',
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        frozen_source_unchanged=True, completed_training={})
        for name, job in payload['jobs'].items():
            r = runner.base.receipt(suite, job)
            if r:
                evidence['completed_training'][name] = dict(run_dir=r['run_dir'], best_epoch=r['best_epoch'],
                    receipt_sha256=runner.filehash(suite/'results'/f'{name}.json'),
                    best_sha256=runner.filehash(Path(r['run_dir'])/'checkpoints/best.pth'),
                    last_sha256=runner.filehash(Path(r['run_dir'])/'checkpoints/last.pth'))
        audit = Path(__file__).parent/'attempts'/f'{time.time_ns()}.json'
        runner.write(audit, evidence)
        original_launch = runner.base.launch
        def protected_launch(target_suite, job, gpu):
            name = job['variant']
            if name in evidence['completed_training']:
                old = evidence['completed_training'][name]
                for filename, key in [('best.pth','best_sha256'),('last.pth','last_sha256')]:
                    if runner.filehash(Path(old['run_dir'])/'checkpoints'/filename) != old[key]:
                        raise RuntimeError('Completed checkpoint changed: '+name)
                if runner.filehash(suite/'results'/f'{name}.json') != old['receipt_sha256']:
                    raise RuntimeError('Completed receipt changed: '+name)
                print('Skip completed training:', name, flush=True)
            return original_launch(target_suite, job, gpu)
        runner.base.launch = protected_launch
        runner.base.evaluate_record = lambda *args, **kwargs:scoped_evaluate_record(runner.base,*args,**kwargs)
        try:
            runner.run_queue(suite, payload)
            evidence.update(status='completed', finished=time.time())
        except BaseException as error:
            evidence.update(status='failed', error=repr(error), finished=time.time())
            runner.write(suite/'state.json',dict(phase='failed',error=repr(error),time=time.time()))
            raise
        finally:
            runner.write(audit,evidence)


if __name__ == '__main__':
    main()
