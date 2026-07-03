#!/usr/bin/env python3
from common import BENCH, config, finish, parse_args, run_stages

def main():
    args = parse_args("E2GAN")
    finish(run_stages("E2GAN", args, BENCH / "E2GAN", [[args.python, "train.py", "--config", config(args, "E2GAN")]]))

if __name__ == "__main__": main()
