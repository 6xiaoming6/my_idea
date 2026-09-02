# V14 root-cause exploration

All commands are run from the repository root.  Every candidate remains a
single-stage, end-to-end V14 training run.  Validation selects one `best.pt`,
which is loaded once for the final test.

## Final model-hyperparameter exploration

The final capacity/routing sweep changes one existing V14 JSON parameter per
candidate. It covers 15 candidates on Core-6 (90 full jobs, about 76 measured
single-GPU hours). It changes no data, loss, or training workflow and does not
inherit any earlier exploration candidate.

```bash
# One-job worst-case pipeline smoke
python scripts/v14-exploration/run_hparam_exploration.py \
  --gpu 0 --phase screen --candidates H03 --epochs 1 --max-jobs 1

# Complete resumable screen; omit --epochs
python scripts/v14-exploration/run_hparam_exploration.py \
  --gpu 0 --phase screen

# Freeze the validation-only decision
python scripts/v14-exploration/summarize_hparam_screen.py
```

Only a candidate passing every frozen condition in
`model_designs/V14最终模型容量与路由超参数探索方案.md` may enter the optional
three-seed Core-6 stage. Do not combine candidates or rank them by test metrics.

## BRLG: bounded regional-local residual gate

BRLG tests whether V14's sample-level scalar final gate should allow bounded
temporal or low-resolution regional modulation. It adds no experts and changes
neither losses nor the main/CTF backbone. Six candidates form a clean
`2 granularities × 3 bounds` matrix and run 36 complete Core-6 jobs (about 27.7
single-GPU hours from recent measurements).

Pipeline smoke:

```bash
python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase screen --candidates B05 --epochs 1
```

Complete Core-6 (omit `--epochs`):

```bash
python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase screen
```

After all 36 jobs finish, freeze the validation-only decision:

```bash
python scripts/v14-exploration/summarize_brlg_screen.py
```

Do not run multiseed/all24 or inspect candidate test rankings before the frozen
Core-6 decision. Full protocol:
`model_designs/V14有界区域级局部残差门BRLG验证实验方案.md`.

## ESAP: evidence-calibrated scale-adaptive pathways

The newest long-form route contains 18 preregistered candidates evaluated on
Core-6, for 108 complete single-GPU jobs. Recent measured runtimes estimate
83.2 hours (3.47 days). It changes only scale-fusion calibration and keeps the
V14 expert pool, Top-K routing, backbone, optimizer, and single-stage training
workflow intact.

Pipeline smoke:

```bash
python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 --phase screen --candidates E01 --epochs 1
```

Complete screen (omit `--epochs`):

```bash
python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 --phase screen
```

The command is resumable and skips complete runs. Candidate selection must use
validation metrics and the frozen rules in
`model_designs/V14证据校准自适应多尺度路径ESAP完整探索方案.md`; do not inspect test
rankings or run multiseed/all24 before freezing the screen decision.

After all 108 jobs finish, generate the validation-only frozen decision:

```bash
python scripts/v14-exploration/summarize_esap_screen.py
```

### ESAP Core-6 decision

The complete 108-job screen is finished and no candidate passes all frozen
promotion rules. E08 has the best validation macro MAE change (-4.078%), but
only 3/6 clear MAE wins, validation macro RMSE regresses by 1.198%, and its
worst point RMSE regression is 8.149%. After freezing the decision, E08 is also
the only candidate with improved test macro MAE (-1.993%), but it wins only 2/6
test points and test macro RMSE still regresses by 1.720%. Do not run ESAP
multiseed/all24 or interpolate more ESAP hyperparameters. Keep original V14 as
the formal base. Full evidence is recorded in
`outputs/v14-exploration/summary/esap_core6_analysis.md`.

## Completed diagnostics

Reproduce the full 24-point residual multiplier and exact-metric audit:

```bash
conda activate difftdi
python scripts/v14-exploration/diagnose_residual_scale.py \
  --gpu 0 \
  --scope all24
```

This command never trains or overwrites a checkpoint.  It selects `kappa` on
validation and evaluates only the selected value, `kappa=0`, and `kappa=1` on
test.

## Root-cause training

First run the recommended screen:

```bash
conda activate difftdi
python scripts/v14-exploration/run_root_cause.py \
  --gpu 0 \
  --phase screen
```

The screen contains:

- E01 on Core-6: remove the final-gate magnitude penalty.
- E02 on two points: weights `1e-5`, `1e-4`, and `1e-3`.
- E03 on two points: weights `0.01`, `0.03`, and `0.05`.

After choosing the best E02/E03 weights from validation, expand them to Core-6:

```bash
python scripts/v14-exploration/run_root_cause.py \
  --gpu 0 \
  --phase core6 \
  --experiments E02 E03 \
  --e02-weight 1e-4 \
  --e03-weight 0.01
```

Only after Core-6 passes the promotion criteria should a candidate run all 24
points.  For example:

```bash
python scripts/v14-exploration/run_root_cause.py \
  --gpu 0 \
  --phase all24 \
  --experiments E02 \
  --e02-weight 1e-4
```

Completed runs are skipped by default.  Use `--rerun-completed` only when an
intentional repeat is required.  Use `--epochs 1` only for a pipeline smoke
test; omitting it uses the complete V14 dataset-specific epoch settings.

## Conditional follow-up experiments

Do not run these candidates together by default.  Their prerequisites are:

- E04: run only if both E02 and E03 pass the complete Core-6 criteria.
- S01: run only if E02 confirms that residual-scale identifiability matters.
- S02: run only if the two-channel multiplier grid shows stable channel-specific
  optima.

Reproduce the channel prerequisite diagnostic:

```bash
python scripts/v14-exploration/diagnose_channel_scale.py --gpu 0
```

The completed 16-point diagnosis selected different multipliers for the two
channels at only 3/16 points, and the preferred direction was not consistent.
Therefore S02 currently fails its prerequisite and must not enter formal
training.

Generic E04 command (do not run for the current results because E03 failed):

```bash
python scripts/v14-exploration/run_followup_candidates.py \
  --gpu 0 \
  --candidates E04
```

Candidates remain independent:

- E04 combines only the validated E02/E03 loss terms.
- S01 changes only the effective-residual parameterization and keeps original
  V14 losses.
- S02 changes only the final gate granularity and keeps the original residual
  parameterization and losses.

The Core-6 decision is now:

1. E02 (`1e-4`) passed: 4/6 MAE wins, mean MAE improvement 1.773%, and all
   safety limits passed.
2. E03 (`0.01`) failed: only 1/5 completed points won and the TaxiBJ
   fixed@0.4 MAE/RMSE regressions exceeded the pointwise limits.
3. E04 is rejected because its E03 prerequisite failed.
4. S01 failed Core-6: only 1/6 MAE wins, with mean MAE/RMSE regressions of
   8.869%/13.003%. It is rejected and must not be combined with E02.
5. S02 remains rejected unless new channel evidence overturns the current
   diagnosis.
6. E02 is the only candidate promoted to multi-seed confirmation.

## Multi-seed confirmation

Run E02 with fixed offline masks and three model seeds:

```bash
python scripts/v14-exploration/run_multiseed.py \
  --gpu 0 \
  --candidates E02 \
  --seeds 42 2026 3407
```

The existing six seed-42 runs are skipped, so the first invocation schedules
12 new full runs. Do not regenerate the random-mask CSV files: this stage
measures optimization randomness, not mask randomness.

The matched V14 and E02 Core-6 model-seed runs are complete:

- all 36 runs and checkpoints are complete, with no NaN/Inf;
- E02 wins all 6/6 three-seed mean MAE and RMSE comparisons;
- macro MAE/RMSE improve by 1.420%/2.021%;
- all three datasets improve on average;
- the maximum paired seed-level MAE regression is 2.875%, below the 3% limit;
- parameter count and peak memory do not increase.

E02 therefore passes model-seed confirmation. Run the seed-42 all-24 stage:

```bash
python scripts/v14-exploration/run_root_cause.py \
  --gpu 0 \
  --phase all24 \
  --experiments E02 \
  --e02-weight 1e-4
```

The six completed E02 Core-6 jobs are skipped, leaving 18 full jobs.

## All-24 final decision

The seed-42 all-24 stage is complete:

- all 24 formal revision-2 runs completed with finite metrics and checkpoints;
- E02 has 18/24 numerical MAE wins, but only 10/24 clear wins of at least 0.5%;
- macro MAE/RMSE improve by only 0.505%/0.360%;
- TaxiBJ random@0.2 regresses by 10.332% MAE and 16.581% RMSE;
- the 0.2-rate group regresses by 1.862% MAE and 2.915% RMSE on average.

E02 fails the all-24 promotion limits (14 clear wins, at most 3% pointwise MAE
regression, and at most 5% pointwise RMSE regression). Do not turn the current
uniform `1e-4` regularizer into a formal model version. Keep V14 as the base and
diagnose smaller weights on the three random@0.2 points before any new all-24
run.

Run the six low-rate diagnostic jobs:

```bash
python scripts/v14-exploration/run_e02_lowrate_diagnosis.py \
  --gpu 0
```

This runs, sequentially on one GPU:

```text
TaxiBJ / BikeNYC / CHAP
× random@0.2
× lambda_v14_delta_scale in {2.5e-5, 5e-5}
× seed 42
= 6 jobs
```

Omit `--epochs` for complete dataset-specific training budgets. Completed jobs
with finite MAE/RMSE, a final test marker, `metrics.jsonl`, and `best.pt` are
skipped automatically. Each job has a four-hour timeout to prevent another
indefinite CUDA/PyTorch hang; rerunning the same command resumes at the first
incomplete job.

The six jobs are complete. Neither small weight is promoted:

- `2.5e-5`: mean MAE/RMSE regress by 6.502%/8.424%, with 0/3 MAE wins;
- `5e-5`: mean MAE/RMSE regress by 1.940%/3.240%, with 1/3 MAE wins;
- the isolated BikeNYC `5e-5` test win is not supported by validation and must
  not be selected after looking at the test set.

Do not extend these weights to more points or seeds, and do not continue
interpolating E02 weights. E02 is closed as a global candidate; retain V14 and
move to a different single-variable, single-stage hypothesis.

## RLB: differentiable Top-K load-aware routing

RLB is independent from the closed E02 residual-scale route. It does not change
the V14 architecture, expert pool, `top_k`, or inference path. The legacy
configuration remains the default; only the new routing experiment configs set
`loss.load_balance_mode` to `switch_topk`.

Run the no-training Stage-0 audit first:

```bash
conda activate difftdi
python scripts/v14-exploration/audit_rlb_routing.py \
  --gpu 0 \
  --scope all24 \
  --split val
```

This reads the existing V14 checkpoints and writes exact routing statistics and
an autograd proof under:

```text
outputs/v14-exploration/diagnostics/rlb_stage0
```

After the audit and unit tests pass, run the nine-job validation screen:

```bash
python scripts/v14-exploration/run_rlb_exploration.py \
  --gpu 0 \
  --phase screen
```

The screen evaluates `1e-4`, `1e-3`, and `1e-2` on TaxiBJ fixed@0.4,
BikeNYC random@0.2, and CHAP random@0.2. Select one candidate using validation
MAE/RMSE and validation routing statistics only. Do not choose it from test
rankings.

If, and only if, the screen passes the preregistered limits in
`model_designs/V14可微TopK负载感知路由完整探索方案.md`, continue in order:

```bash
# Example only: replace R02 with the validation-selected candidate.
python scripts/v14-exploration/run_rlb_exploration.py \
  --gpu 0 \
  --phase core6 \
  --candidate R02

python scripts/v14-exploration/run_rlb_exploration.py \
  --gpu 0 \
  --phase multiseed \
  --candidate R02 \
  --seeds 42 2026 3407

python scripts/v14-exploration/run_rlb_exploration.py \
  --gpu 0 \
  --phase all24 \
  --candidate R02
```

Completed experiments are skipped after checking the experiment metadata,
finite test MAE/RMSE, `metrics.jsonl`, and `best.pt`. Use `--epochs 1` only for
a pipeline smoke test. Omitting `--epochs` uses the full dataset-specific V14
budgets.

### RLB screen decision

The complete nine-job screen is finished. All jobs are complete and finite,
but no candidate passes the preregistered Stage-1 safety limits:

- R01 (`1e-4`) and R02 (`1e-3`) regress TaxiBJ validation MAE by more than 9%.
- R03 (`1e-2`) improves the three-point mean validation MAE by 1.005%, but
  regresses CHAP validation MAE by 1.997%, above the 1% pointwise limit.
- After freezing the validation decision, R03 test MAE improves TaxiBJ but
  regresses BikeNYC and CHAP; its three-point test macro MAE/RMSE regress by
  0.106%/0.225%.

Do not run R01/R02/R03 on Core-6, multiple seeds, or all 24 points. Do not tune
additional weights from these test results. The full decision is recorded in
`outputs/v14-exploration/summary/rlb_screen_summary.md`.

## MCG: bounded monotonic consistency gate

MCG starts from the original V14 again; it does not inherit E02, S01, S02, or
RLB. The read-only 24-point diagnosis is reproduced with:

```bash
python scripts/v14-exploration/diagnose_gate_calibration.py \
  --gpu 0 \
  --scope all24
```

The existing gate alpha is negatively correlated with the missing-region
analytic oracle alpha on all 24 points (mean Spearman `-0.539`). In contrast,
the target-free observed CTF gain is positively correlated with the oracle on
22/24 points. MCG therefore changes only the final gate mapping: the context
MLP is bounded, while the relative observed gain enters with a fixed positive
coefficient. The main backbone, C2F refiner, losses, parameter count, and
single-stage training protocol remain unchanged.

Run the nine-job full-epoch screen:

```bash
python scripts/v14-exploration/run_mcg_exploration.py \
  --gpu 0 \
  --phase screen
```

The screen evaluates gains `0.5`, `1.0`, and `2.0` on TaxiBJ fixed@0.8,
BikeNYC random@0.2, and CHAP random@0.4. TaxiBJ fixed@0.8 is deliberately a
known adverse-correlation stress point; it must not be removed to make the
aggregate result look better. Select a candidate using validation MAE/RMSE
only.

Only after the screen passes the limits in
`model_designs/V14单调一致性门控完整探索方案.md` should the same candidate be
expanded in this order:

```bash
# Example only; replace M01 with the validation-selected candidate.
python scripts/v14-exploration/run_mcg_exploration.py \
  --gpu 0 --phase core6 --candidate M01

python scripts/v14-exploration/run_mcg_exploration.py \
  --gpu 0 --phase multiseed --candidate M01 --seeds 42 2026 3407

python scripts/v14-exploration/run_mcg_exploration.py \
  --gpu 0 --phase all24 --candidate M01
```

The runner skips complete jobs by checking experiment metadata, epoch budget,
finite test MAE/RMSE, `metrics.jsonl`, and `best.pt`. Omitting `--epochs` uses
the formal TaxiBJ/BikeNYC/CHAP budgets of 160/140/150 epochs.

### MCG screen decision

The complete nine-job screen is finished. All formal runs completed with
finite metrics and checkpoints, but no candidate passes the preregistered
validation limits:

- M01/M02/M03 validation macro MAE regress by 0.876%/0.304%/0.429%;
- validation macro RMSE regresses by 7.037%/6.296%/6.686%;
- the maximum pointwise validation RMSE regressions are
  12.391%/10.927%/9.575%;
- after freezing the rejection, test macro MAE also regresses by
  6.672%/5.442%/6.080%.

Do not run M01/M02/M03 on Core-6, multiple seeds, or all 24 points, and do not
interpolate additional gains. The full decision is recorded in
`outputs/v14-exploration/summary/mcg_screen_summary.md`.

## GSA: gradient-scaled stage supervision

GSA starts from original V14 and changes only `lambda_v14_mid` and
`lambda_v14_coarse` by one shared scale. It does not inherit any prior failed
candidate. Reproduce the read-only all-24 gradient audit with:

```bash
python scripts/v14-exploration/diagnose_multiscale_gradients.py \
  --gpu 0 --scope all24 --batches 3
```

The audit finds 10/23 mid-gradient conflict points and 4/5 coarse-gradient
conflict points. The weighted stage gradients are on average 471.39x (mid) and
31.06x (coarse) the final-objective gradient on V14 refiner/controller
parameters.

Run the nine-job full-epoch screen:

```bash
python scripts/v14-exploration/run_gsa_exploration.py \
  --gpu 0 --phase screen
```

G01/G02/G03 use shared stage-supervision scales `0`, `1e-3`, and `1e-2` on
TaxiBJ fixed@0.4, BikeNYC random@0.8, and CHAP fixed@0.8. BikeNYC is an
intentional positive-gradient safety point. Candidate selection must use
validation metrics only. Full criteria and later commands are in
`model_designs/V14多尺度辅助监督梯度缩放完整探索方案.md`.

### GSA screen decision

The nine formal runs are complete and finite, but every candidate fails:

- validation macro MAE regresses by 3.402%/4.823%/4.267% for G01/G02/G03;
- validation macro RMSE regresses by 11.508%/14.440%/14.900%;
- TaxiBJ fixed@0.4 validation MAE regresses by 17.855%–20.992%;
- after freezing rejection, test macro MAE regresses by
  9.054%/10.521%/10.025%.

Do not promote G01/G02/G03 or interpolate more fixed scales. The gradient
conflict is a valid diagnosis, but globally attenuating stage supervision
removes optimization guidance required by TaxiBJ. The complete decision is in
`outputs/v14-exploration/summary/gsa_screen_summary.md`.

## CSAS: continuous stage-auxiliary supervision annealing

CSAS follows the GSA failure without inheriting its fixed weak weights. It
keeps the original V14 mid/coarse weights early, then smoothly cosine-decays
their shared scale to zero. This remains one continuous end-to-end training
run and does not change architecture or inference.

Run the nine full-epoch screen:

```bash
python scripts/v14-exploration/run_csas_exploration.py \
  --gpu 0 --phase screen
```

C01/C02/C03 begin decay at 0%, 25%, and 50% of training. The screen reuses
TaxiBJ fixed@0.4, BikeNYC random@0.8, and CHAP fixed@0.8 for a direct GSA
comparison. Selection must use validation metrics only. Full definitions and
promotion criteria are in
`model_designs/V14连续阶段辅助监督退火完整探索方案.md`.

### CSAS screen decision

All nine formal runs completed with finite metrics. C01 and C02 fail clearly.
C03 is directionally useful but still fails the preregistered safety limits:

- C03 validation macro MAE improves by 3.355% with 2/3 wins;
- validation macro RMSE regresses by 2.372%;
- BikeNYC validation MAE/RMSE regress by 1.747%/6.568%;
- after freezing the decision, test macro MAE improves by 1.042%, but CHAP
  test RMSE regresses by 8.459%.

Do not promote C01/C02/C03 to Core-6 and do not interpolate more decay start
fractions after viewing these results. C03 remains mechanism evidence that
early full supervision plus late attenuation is better than fixed attenuation.
The complete decision is in
`outputs/v14-exploration/summary/csas_screen_summary.md`.
