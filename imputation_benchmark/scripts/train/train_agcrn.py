#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("AGCRN")
    finish(run_stages("AGCRN", args, BASELINES / "AGCRN", [[args.python, "model/Run.py", "--config", config(args, "AGCRN")]]))

if __name__ == "__main__": main()
