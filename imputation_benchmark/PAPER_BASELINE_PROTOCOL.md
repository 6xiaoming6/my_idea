# Paper baseline comparison protocol

## Default experiment matrix

- Datasets: TaxiBJ, BikeNYC, CHAP
- Masks: fixed, random
- Missing rates: 0.2, 0.4, 0.6, 0.8
- Baselines: 13 implemented mainline methods (MeanFill, HistoricalAverage,
  LATC, BRITS, GAIN, CSDI, SAITS, GRIN, PriSTI, ImputeFormer, STCPA,
  STAMImputer, PAST)
- Seed: 42, matching the current main-model experiments
- Channel: 0, matching the validated baseline protocol
- Total: 312 runs

The eight methods listed in `EXTRA_BASELINES.md` are appendix candidates and
are disabled in all default launchers and policies.

Train/validation/test boundaries are derived from the original project NPZ
files, not a generic 60/20/20 split. Every method receives the same mask file
for a given dataset/pattern/rate. Test results are produced only after model
selection by validation data.

For a stronger statistical paper, rerun both the proposed model and baselines
with seeds 42, 43, and 44 and report mean ± standard deviation. Do not report
three-seed baseline results against a single-seed proposed-model result.

## Training policy

Network architecture and the optimizer/learning-rate definitions remain those
of each baseline. Training budgets and batches are normalized for a fair and
feasible comparison on 24 GB GPUs.

| Model | Training budget | Batch, 1024 nodes | Batch, 288 nodes | `val_epoch` |
|---|---:|---:|---:|---:|
| BRITS | 100 epochs | 32 | 64 | 2 |
| CSDI | 100 epochs | 4 | 16 | 10 |
| GAIN | 100 epochs | 32 | 64 | 2 |
| ImputeFormer | 100 epochs | 8 | 32 | 5 |
| PriSTI | 100 epochs | 2 | 8 | 10 |
| LATC | 100 iterations, original algorithm | — | — | — |
| SAITS | 100 epochs | 4 | 16 | 5 |
| GRIN | 100 epochs | 2 | 8 | 5 |
| STCPA | 100 epochs | 4 | 16 | 5 |
| STAMImputer | 100 epochs | 1 | 4 | 10 |
| PAST | 50 epochs | 2 | 8 | 5 |
| MeanFill | non-neural, one fit/evaluation | — | — | — |
| HistoricalAverage | non-neural, one fit/evaluation | — | — | — |

Batch size affects throughput and optimizer noise but not model structure. The
smaller 1024-node batches prevent OOM without shrinking any network layer.
Validation cadence is selected from the actual validation workload: models
with many small batches use 10, medium workloads use 5, and light workloads
use 2. For models with early stopping, `patience` counts validation events and
is adjusted so the paper policy still represents roughly 20 epochs without an
improvement (for example, `val_epoch=5`, `patience=4`). The final epoch is
always validated regardless of cadence.

## Metrics

- Report MAE and RMSE for every dataset.
- Validation and test MAE/RMSE are computed after inverse transformation in
  each dataset's original value range, and only on artificially hidden entries
  (`mask == 0`). Normalized training losses remain available as optimization
  diagnostics but must not be used in the paper comparison table.
- Report MAPE for CHAP only. TaxiBJ and BikeNYC contain values near zero, so
  their MAPE is numerically unstable and should not be used for conclusions.
- Preserve `raw.log` and the generated config for reproducibility.
- `val_epoch` in the selected JSON policy controls validation cadence; the
  final epoch is always validated. Only validation improvements overwrite the
  single best checkpoint, which is loaded for one final test evaluation.
- Preserve `train.log`, `val.log`, and `test.log` for every run. ImputeFormer's
  local training scheduler now honors the same `val_epoch` policy without
  changing its PyPOTS model architecture. LAST and LATC have no conventional
  validation epoch.
- Each trainable baseline keeps exactly one validation-selected best checkpoint
  under its standardized run directory. Better states replace earlier ones and
  superseded files are removed after testing. Add `--no-checkpoint` to retain
  only logs and metrics. LAST and LATC are non-parametric and have no weights.

## Launcher

```bash
python imputation_benchmark/run_all_baseline_train_2gpu.py --gpus 0 1
```

To split the two mask families across two independently visible terminals, run:

```bash
# Terminal 1: GPU 0 runs every fixed experiment
python imputation_benchmark/run_all_baseline_train_2gpu.py \
  --gpus 0 \
  --masks fixed

# Terminal 2: GPU 1 runs every random experiment
python imputation_benchmark/run_all_baseline_train_2gpu.py \
  --gpus 1 \
  --masks random
```

The launchers use separate run-summary directories, while model checkpoints are
scoped by dataset and mask so the two processes cannot prune each other's best
weights.

## JSON-controlled five-epoch integration test

Training budgets, batch sizes, patience, fine-tuning allocation, and validation
intervals are controlled by JSON files under `imputation_benchmark/policies/`.
Run every baseline once on a representative TaxiBJ/fixed/0.2 protocol with:

```bash
python imputation_benchmark/run_all_baseline_train_2gpu.py \
  --gpus 0 \
  --datasets TaxiBJ \
  --masks fixed \
  --rates 0.2 \
  --policy-json imputation_benchmark/policies/baseline_5epoch_test.json
```

This launches 13 jobs. Ten trainable neural baselines run five epochs and
complete validation, best-checkpoint selection, and final testing; LATC runs
five iterations, while MeanFill and HistoricalAverage fit training-set
statistics once and evaluate without fabricated epochs. The default policy is
`baseline_paper.json`; switching back to formal training requires no source-code edits.

Resume an interrupted run using the directory printed by the launcher:

```bash
python imputation_benchmark/run_all_baseline_train_2gpu.py \
  --gpus 0 1 \
  --resume-run imputation_benchmark/paper_runs/<run_id> \
  --skip-prepare
```

Use `--dry-run` to inspect the complete matrix and `--prepare-only` to prepare
all datasets/configurations without occupying GPUs.

Model stdout is shown live with a GPU/job prefix and simultaneously written to
`launcher_logs`. Add `--quiet-console` only when running under an external job
scheduler. Failed jobs print their return code and final log lines immediately;
resume attempts use `.attemptN.log` and preserve the earlier failure log.

Each launch also writes `manifest.json`, `summary.json`, `summary.md`, and one
live `.log` per job under
`imputation_benchmark/paper_runs/<run_id>/launcher_logs/`. Standardized model
records are stored under
`outputs/<dataset>/baseline/<model>/<mask>/rate<rate>/<run_id>/logs/` as
`train.log`, `val.log`, `test.log`, and `raw.log` (plus `metrics.jsonl`).
The standardized logs explicitly record `metric_scale` and `metric_scope`;
`val.log` stores original-scale validation MAE/RMSE at each configured
validation epoch, while `test.log` stores the one final result from the
validation-selected best model.
