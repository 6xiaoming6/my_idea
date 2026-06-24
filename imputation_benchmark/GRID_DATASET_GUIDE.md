# Grid-dataset baseline adapter

`prepare_grid_dataset.py` is the sole data-adaptation layer for the benchmark.
It leaves all baseline model definitions untouched.

It converts the project's window NPZ layout `[samples, C, T, H, W]` to the
benchmark layout.  The canonical arrays are `split_*` arrays shaped
`[samples, time, nodes]`, which preserve the project's existing train/val/test
samples exactly.  A legacy compatibility stream is also supplied as `data`
with layout `[time, nodes, features]`; it concatenates complete windows and is
only suitable for legacy loaders that cannot consume the split arrays.  With
the default `--channel all`,
`nodes = C * H * W` and `features = 1`; this is important because many legacy
baseline loaders only read feature zero.  No inflow/outflow channel is dropped.

Run from the project root (`my_idea`):

```bash
python imputation_benchmark/prepare_grid_dataset.py \
  --dataset TaxiBJ --mask fixed --rate 0.4
```

Outputs are under `imputation_benchmark/data/adapted/<dataset>/<mask>_<rate>/channel_all/`:

- `true_data_<mask>_<rate>_v2.npz`: canonical split data/masks (stored once)
- `miss_data_<mask>_<rate>_v2.npz`: compact format marker, paired with the true file
- `grid_edges.csv`: directed four-neighbour grid graph (and same-cell cross-channel links)
- `manifest.json`: dimensions, realised missing rate, source paths, and the exact split ratios to use

The `fixed` and `random` modes consume the exact mask CSVs already used by the
main model.  They are exact in the canonical `split_*` arrays.  The legacy
stream repeats a source mask across the source window; do not use that stream
to claim bit-for-bit equivalence for a random-mask experiment.

The adapter can also create the benchmark's four standard scenarios without
touching `GenerateData/generator.py` or requiring its unpinned Louvain package:

```bash
python imputation_benchmark/prepare_grid_dataset.py \
  --dataset BikeNYC --mask SC-TC --rate 0.4 --seed 42
```

## Model configuration

Point a baseline's `data_prefix` at the output directory and use the adapter
mask name/rate.  Prefer its split-aware data loader; CSDI has already been made
split-aware.  A legacy loader that only understands `data` must use the
manifest's `val_ratio` and `test_ratio` rather than the PEMS defaults (usually
0.2/0.2).  Set `num_of_vertices`/`num_nodes` to `nodes`, and use
`grid_edges.csv` wherever the model asks for an adjacency/distance CSV.

For an unmodified legacy loader that can only read a continuous `data` array,
generate a separate compatibility output with `--legacy-stream`; its
`true_data_*` and `miss_data_*` names are exactly what BRITS, ASTGNN, GCASTN,
SSTBAN, LAST and LATC expect.  Use a separate `--output-root` from the
split-aware output.

For the TaxiBJ and BikeNYC two-channel data, `--channel all` is the correct
form for sequence models such as CSDI, mTAN, GAIN, E2GAN, BRITS and
ImputeFormer.  Dense graph-attention models can be prohibitively large at
TaxiBJ's 2048 nodes.  For those models run each flow channel independently:

```bash
python imputation_benchmark/prepare_grid_dataset.py \
  --dataset TaxiBJ --mask fixed --rate 0.4 --channel 0
python imputation_benchmark/prepare_grid_dataset.py \
  --dataset TaxiBJ --mask fixed --rate 0.4 --channel 1
```

Report the element-count-weighted aggregate of the two test metrics.  This is a
data-layout choice, not a model-architecture change.

All future baselines should consume this same canonical NPZ + manifest contract;
they need no dataset-specific loader branch.

## Adapted benchmark loaders

The following existing loaders now consume `split_*` arrays directly and keep
the original project split intact: **CSDI, GAIN, E2GAN, mTAN, ImputeFormer,
AGCRN, IGNNK, and PriSTI**.  Their network modules are unchanged.  The graph models
use `grid_edges.csv`; use a single channel for dense-attention graph models on
TaxiBJ when GPU memory requires it.

The remaining historical implementations (BRITS, PriSTI, ASTGNN, GCASTN,
SSTBAN, LAST, LATC) use their own additional sample-generation formats.  They
should be migrated through the same adapter contract rather than through a
model change; do not feed them the old PEMS paths directly.

## First runnable baseline: CSDI on TaxiBJ fixed 0.4

The repository includes a split-aware CSDI loader and ready configuration.  On
the machine's CUDA/PyTorch environment:

```bash
cd my_idea
python imputation_benchmark/prepare_grid_dataset.py \
  --dataset TaxiBJ --mask fixed --rate 0.4 --channel 0
cd imputation_benchmark/CSDI
conda run -n difftdi python train.py --config config/TaxiBJ_channel0_fixed_0.4.conf
```

It trains and then prints test MAE/RMSE/MAPE on exactly the adapted test split.
The CSDI diffusion model itself is unchanged.  TaxiBJ's 2048 flattened nodes
make full feature attention expensive, so the ready configuration runs channel
zero (1024 nodes) with the original four layers and batch size one.  Generate
and run the matching `--channel 1` dataset/config, then aggregate its test
errors with channel zero by element count.
