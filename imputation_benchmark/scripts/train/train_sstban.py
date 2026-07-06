#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("SSTBAN")
    cwd = BASELINES / "SSTBAN" / "SSTBAN-imputation"
    finish(run_stages("SSTBAN", args, cwd, [[args.python, "train_SSTBAN.py", "--config", config(args, "SSTBAN")]]))

if __name__ == "__main__": main()
