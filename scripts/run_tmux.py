#!/usr/bin/env python3
"""Start a dedicated session whose main process exits with train + evaluation."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--session', default='i37-release')
p.add_argument('--gpu', type=int, default=1)
a = p.parse_args()
a.run_dir.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--data-dir', str(a.data_dir.resolve()),
           '--run-dir', str(a.run_dir.resolve()), '--gpu', str(a.gpu), '--evaluate']
# pipefail preserves failures while tee keeps output visible and saved. No idle shell remains.
body = 'set -o pipefail; ' + shlex.join(command) + ' 2>&1 | tee -a ' + shlex.quote(str(a.run_dir.resolve() / 'console.log'))
subprocess.run(['tmux', 'new-session', '-d', '-s', a.session, shlex.join(['bash', '-c', body]),
                ';', 'set-option', '-w', '-t', a.session, 'remain-on-exit', 'off'], check=True)
print('tmux attach -t ' + shlex.quote(a.session))
