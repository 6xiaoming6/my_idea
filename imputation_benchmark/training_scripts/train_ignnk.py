#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("IGNNK")
    finish(run_stages("IGNNK", args, BENCH / "IGNNK", [[args.python, "train.py", "--config", config(args, "IGNNK")]]))

if __name__ == "__main__": main()
