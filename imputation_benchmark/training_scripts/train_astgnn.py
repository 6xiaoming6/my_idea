#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("ASTGNN")
    finish(run_stages("ASTGNN", args, BENCH / "ASTGNN", [[args.python, "train_ASTGNN.py", "--config", config(args, "ASTGNN"), "--cuda", "0"]]))

if __name__ == "__main__": main()
