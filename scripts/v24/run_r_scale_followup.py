#!/usr/bin/env python3
"""R11/R12: R7 scale soft start and persistent two-scale weighted fusion."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_r_exploration as previous
from stmoe_imputer.config import deep_update

VARIANTS=('R11','R12')
CONFIG_DIR=ROOT/'configs/v24/r_scale_followup'


def jobs(dataset='taxibj',epochs=100,batch_size=32,variants=VARIANTS):
    result=[]
    for variant in variants:
        spec=previous.baseline.load(CONFIG_DIR/f'{variant}.json')
        job=previous.jobs(dataset,epochs,batch_size,(spec['base_variant'],))[0]
        cfg=deep_update(copy.deepcopy(job['config']),spec['override'])
        cfg['experiment_plan'].update(suite='r_scale_followup',variant=variant,base_variant=spec['base_variant'])
        result.append({**job,'variant':variant,'name':spec['name'],'config':cfg})
    return result


def manifest(dataset):
    source=previous.manifest(dataset)
    for p in [*CONFIG_DIR.glob('*.json'),ROOT/'scripts/v24/README_R_SCALE_FOLLOWUP.md',
              ROOT/'tests/test_v24_r_scale_followup.py']:
        source[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return source


def main():
    # Existing R7/R5/N5 results remain pinned in their original frozen suite.
    # This queue trains only these two new variants, followed by the same six tests.
    previous.main(job_builder=jobs,source_manifest=manifest,reference_builder=lambda dataset:[],
                  variants=VARIANTS,suite_name='r_scale_followup',description=__doc__)

if __name__=='__main__':main()
