# Full baseline training scripts

These scripts train on the complete adapted dataset. They are separate from
`smoke_tests`: no window limit, no forced one-epoch setting, and no forced
smoke batch size. Model architecture and training hyperparameters come from
each baseline's original template.

Example:

```bash
python imputation_benchmark/training_scripts/train_astgnn.py \
  --dataset TaxiBJ --mask fixed --rate 0.2 --channel 0 --gpu 0
```

The first run prepares full data under `imputation_benchmark/data/adapted` and
generates a config under `imputation_benchmark/training_configs`. Reuse them:

```bash
python imputation_benchmark/training_scripts/train_astgnn.py \
  --dataset TaxiBJ --mask fixed --rate 0.2 --channel 0 --gpu 0 --no-prepare
```

`--timeout 0` (the default) means no training timeout. Use `--channel all` to
train on all source channels as graph nodes; the default is channel `0`, which
matches the validated baseline protocol.

## Standardized logs

Every training script keeps the model's untouched console output and also
creates the same readable/machine-readable log set as the main project:

```text
outputs/<dataset>/baseline/<model>/<mask>/rate<rate>/
  <timestamp>_seed<seed>_bs<batch>/
    config.json
    logs/
      train.log
      val.log
      test.log
      metrics.jsonl
      raw.log
    checkpoints/
      best_model.<native extension>
    result.json
```

`train.log`, `val.log`, and `metrics.jsonl` use the main project's columns and
JSON schema. `raw.log` preserves all original baseline output for auditing.
When an original baseline does not report a particular metric, the readable
table records `n/a` and JSONL records `null`; no value is fabricated. Override
the root with `--output-root /path/to/outputs`.

By default, every trainable baseline retains exactly one checkpoint selected by
its validation criterion. A newly better state overwrites the prior candidate;
after final testing, any superseded native checkpoints are removed and the best
one is stored under `checkpoints/`. Use `--no-checkpoint` to retain logs and
metrics only. LAST and LATC are non-parametric algorithms, so they do not
produce a model checkpoint.

Each JSON model entry has `val_epoch`, meaning validation runs every N training
epochs (and always after the final epoch). Early stopping and best-checkpoint
updates happen only on validation epochs. After training, the selected best
checkpoint is loaded and the test split is evaluated once; its metrics are
written to `test.log`. ImputeFormer uses a local PyPOTS-compatible training scheduler and therefore
honors `val_epoch` without changing its model architecture. LAST and LATC have
no conventional train/validation loop.

For the two-GPU full paper matrix, use
`imputation_benchmark/run_all_baseline_train_2gpu.py`. The exact fairness
protocol and model-specific batch budgets are documented in
`imputation_benchmark/PAPER_BASELINE_PROTOCOL.md`.

The launcher also accepts a single worker GPU. This supports running all
`fixed` jobs on GPU 0 and all `random` jobs on GPU 1 in separate terminals;
see the protocol document for the exact commands.

Use `--policy-json imputation_benchmark/policies/baseline_5epoch_test.json` for
the five-epoch end-to-end test. Omitting the option selects
`baseline_paper.json`. The JSON policy is translated into each upstream
model's required INI/YAML format; model architecture and optimizer definitions
remain in the original baseline templates.
