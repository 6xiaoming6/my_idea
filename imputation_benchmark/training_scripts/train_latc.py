#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("LATC")
    finish(run_stages("LATC", args, BENCH / "LATC", [[args.python, "train_LATC.py", "--config", config(args, "LATC")]]))

if __name__ == "__main__": main()
