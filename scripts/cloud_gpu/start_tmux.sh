#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then echo "Usage: $0 SESSION [run.py options]" >&2; exit 2; fi
session=$1
shift
[[ "$session" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid session name' >&2; exit 2; }
command -v tmux >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then echo 'Session already exists' >&2; exit 1; fi
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
python_bin=${PYTHON:-python}
suite=$("$python_bin" "$root/scripts/cloud_gpu/run.py" "$@" --prepare-only)
mkdir -p "$suite/launcher_logs"
# Forward a resume GPU override as well as the immutable plan's default GPU.
gpu_args=()
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
    if [[ ${args[i]} == --gpu ]]; then gpu_args=(--gpu "${args[i+1]}"); fi
    if [[ ${args[i]} == --gpu=* ]]; then gpu_args=("${args[i]}"); fi
done
printf -v command_line '%q ' "$python_bin" -u "$root/scripts/cloud_gpu/run.py" --resume "$suite" "${gpu_args[@]}"
printf -v log_path '%q' "$suite/launcher_logs/session.log"
printf -v shell_command 'bash -o pipefail -c %q' "$command_line 2>&1 | tee -a $log_path"
tmux new-session -d -s "$session" "$shell_command"
printf 'Session: %s\nSuite: %s\nAttach: tmux attach -t %s\n' "$session" "$suite" "$session"
