#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, rate_label, run_stages

def setup(args, cwd):
    tag = f"{args.dataset}_{args.mask}_{rate_label(args.rate)}_channel_{args.channel}"
    (cwd / "training_data" / tag).mkdir(parents=True, exist_ok=True)
    (cwd / "experiments" / tag).mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args("BRITS")
    finish(run_stages("BRITS", args, BASELINES / "BRITS", [
        [args.python, "input_process.py", "--config", config(args, "BRITS_prepare")],
        [args.python, "main.py", "--config", config(args, "BRITS")],
    ], setup=setup))

if __name__ == "__main__": main()
