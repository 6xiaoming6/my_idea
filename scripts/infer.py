#!/usr/bin/env python3
"""Complete supplied visible observations and binary masks, without target arrays."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='NPZ with x_{f,m,c}_obs and m_{f,m,c}')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.batch_size < 1: raise ValueError('batch-size must be positive')
    if a.device == 'cuda': os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import configure
    torch.set_num_threads(4)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    configure(checkpoint['config'])
    model = DualBranchSTImputer.from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(a.device).eval()
    with np.load(a.input, allow_pickle=False) as z:
        inputs = {k: np.array(z[k], dtype=np.float32) for s in ('f', 'm', 'c') for k in ('x_'+s+'_obs', 'm_'+s)}
    n = len(inputs['x_f_obs'])
    if not n or any(len(x) != n for x in inputs.values()): raise ValueError('Inconsistent or empty batch')
    predictions = []
    with torch.inference_mode():
        for i in range(0, n, a.batch_size):
            batch = {k: torch.from_numpy(x[i:i+a.batch_size]).to(a.device) for k, x in inputs.items()}
            predictions.append(model(batch)['x_comp'].cpu().numpy())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('wb') as out: np.savez_compressed(out, F_complete=np.concatenate(predictions))


if __name__ == '__main__': main()
