#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("CSDI")
    finish(run_stages("CSDI", args, BASELINES / "CSDI", [[args.python, "train.py", "--config", config(args, "CSDI")]]))

if __name__ == "__main__": main()
