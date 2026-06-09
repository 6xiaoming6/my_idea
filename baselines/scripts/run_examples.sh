#!/usr/bin/env bash
set -e
CONFIG="configs/example_taxibj.json"
python eval_statistical.py --config $CONFIG --method mean
python eval_statistical.py --config $CONFIG --method historical
python eval_statistical.py --config $CONFIG --method linear
python train_deep.py --config $CONFIG --model fine_only
python train_deep.py --config $CONFIG --model conv3d_unet
python train_deep.py --config $CONFIG --model transformer
python train_deep.py --config $CONFIG --model ms_concat
python train_deep.py --config $CONFIG --model fixed_experts
python train_deep.py --config $CONFIG --model no_router
