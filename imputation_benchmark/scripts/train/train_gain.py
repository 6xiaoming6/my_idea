#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("GAIN")
    finish(run_stages("GAIN", args, BASELINES / "GAIN", [[args.python, "train.py", "--config", config(args, "GAIN")]]))

if __name__ == "__main__": main()
