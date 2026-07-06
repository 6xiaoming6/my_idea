#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, rate_label, run_stages

def main():
    args = parse_args("LAST")
    finish(run_stages("LAST", args, BASELINES / "LAST", [[
        args.python, "main.py", "--config", config(args, "LAST"),
        "--miss_type", args.mask, "--miss_rate", rate_label(args.rate),
    ]]))

if __name__ == "__main__": main()
