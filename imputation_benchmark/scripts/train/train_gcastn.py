#!/usr/bin/env python3
from common import BASELINES, config, finish, parse_args, run_stages

def main():
    args = parse_args("GCASTN")
    cwd = BASELINES / "GCASTN" / "GCASTN-main" / "code_data_paper_632" / "GCASTN"
    finish(run_stages("GCASTN", args, cwd, [[args.python, "train_GCASTN.py", "--config", config(args, "GCASTN"), "--cuda", "0"]]))

if __name__ == "__main__": main()
