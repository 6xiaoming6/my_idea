# Spatio-Temporal Imputation Baselines

用于 TaxiBJ / BikeNYC 这类网格时空数据补全任务的基础 baseline 代码包。

## 已实现 baseline

统计类：
- Mean Fill
- Historical Average
- Linear Interpolation

深度模型：
- FineOnlyConv3D：单尺度 Conv3D 补全
- Conv3DUNet：单尺度 3D U-Net
- SimpleTransformerImputer：展平成多变量时间序列的 Transformer baseline

多尺度消融：
- MultiScaleConcatFusion：fine/mid/coarse 普通 concat 融合
- FixedScaleExperts：一尺度一专家，固定尺度专家
- SharedExpertsNoRouter：共享专家池但没有动态 router

可选：
- SAITS/PyPOTS wrapper，需要 `pip install pypots`

## 数据格式

推荐先把数据整理成：

```text
data.npy: [T_total, C, H, W]
```

TaxiBJ 通常是：

```text
[T_total, 2, 32, 32]
```

训练窗口内部使用：

```text
x_f_gt:  [B, C, T, H, W]
m_f:     [B, 1, T, H, W]
x_f_obs: [B, C, T, H, W]
```

mid/coarse 由 fine observed data 和 mask 通过 masked average pooling 构造，避免数据泄漏：

```text
x_m_obs, m_m = MaskedPool(x_f_obs, m_f)
x_c_obs, m_c = MaskedPool(x_m_obs, m_m)
```

## LibCity TAXIBJ 转 npy

如果你下载的是 LibCity 的 TAXIBJ：

```text
TAXIBJ2013.grid
TAXIBJ2014.grid
TAXIBJ2015.grid
TAXIBJ2016.grid
```

运行：

```bash
python -m st_impute.data.parse_libcity_taxibj \
  --data_dir /path/to/TAXIBJ \
  --save_path /path/to/taxibj_flow.npy
```

## 运行统计 baseline

```bash
python eval_statistical.py --config configs/example_taxibj.json --method mean
python eval_statistical.py --config configs/example_taxibj.json --method historical
python eval_statistical.py --config configs/example_taxibj.json --method linear
```

## 运行深度 baseline

```bash
python train_deep.py --config configs/example_taxibj.json --model fine_only
python train_deep.py --config configs/example_taxibj.json --model conv3d_unet
python train_deep.py --config configs/example_taxibj.json --model transformer
python train_deep.py --config configs/example_taxibj.json --model ms_concat
python train_deep.py --config configs/example_taxibj.json --model fixed_experts
python train_deep.py --config configs/example_taxibj.json --model no_router
```

## 缺失模式

支持：

```text
random
spatial_block
temporal_block
spatiotemporal_block
```

示例：

```bash
python train_deep.py --config configs/example_taxibj.json --model ms_concat --missing_type spatial_block --missing_rate 0.6
```

## Config

修改 `configs/example_taxibj.json` 里的 `data_path` 即可开始。
