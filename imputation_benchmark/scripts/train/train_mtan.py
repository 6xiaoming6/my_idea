#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("mTAN")
    finish(run_stages("mTAN", args, BASELINES / "mTAN", [[args.python, "train.py", "--config", config(args, "mTAN")]]))

if __name__ == "__main__": main()
