#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("PriSTI")
    cwd = BENCH / "PriSTI" / "PriSTI-main"
    finish(run_stages("PriSTI", args, cwd, [[
        args.python, "exe_survey.py", "--config", config(args, "PriSTI", "yaml"),
        "--device", "cuda:0", "--num_workers", "0", "--targetstrategy", "random",
    ]]))

if __name__ == "__main__": main()
