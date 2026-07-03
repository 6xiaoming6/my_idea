#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("CSDI")
    finish(run_stages("CSDI", args, BENCH / "CSDI", [[args.python, "train.py", "--config", config(args, "CSDI")]]))

if __name__ == "__main__": main()
