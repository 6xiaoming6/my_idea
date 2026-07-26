# V19-single

V19 is trained end-to-end in one stage. It does not load or freeze a V14
checkpoint.

One-epoch smoke on the three datasets:

```bash
python scripts/v19-single/run_smoke.py --gpu 0
```

Complete 24-run grid, ordered as all fixed runs followed by all random runs:

```bash
python scripts/v19-single/run_full_24.py --gpu 0
```

One formal combination:

```bash
python scripts/v19-single/train.py \
  --dataset TaxiBJ \
  --mask fixed \
  --rate 0.4 \
  --gpu 0
```

Outputs are written below `outputs/v19-single/`.

