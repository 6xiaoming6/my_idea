"""
CHAP PM2.5 NetCDF → NPZ 预处理脚本

将 CHAP 每日 1km PM2.5 数据先裁切到北京地区，再选取固定子区域并构造滑动窗口，
输出标准 [N, C, T, H, W] 格式 NPZ 文件。

用法:
  python scripts/prepare_chap_npz.py                          # 默认: 北京 32x32 子区域, stride=1
  python scripts/prepare_chap_npz.py --window 7 --stride 3   # 7天窗口, 步长3
  python scripts/prepare_chap_npz.py --region bj --years 2020 2021
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# conda run 环境可能缺 h5netcdf；fallback 到 scipy（仅 NetCDF3）或 netCDF4
try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# 预设区域 (lat_min, lat_max, lon_min, lon_max)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "bj": (39.40, 41.10, 115.40, 117.60),  # 北京
    "bth": (38.00, 41.50, 114.00, 119.00),  # 京津冀
    "yrd": (30.00, 33.00, 118.00, 122.00),  # 长三角
}


def _find_indices(lat: np.ndarray, lon: np.ndarray, region: tuple[float, float, float, float]):
    """返回裁切索引并调整到 H, W 均可被 4 整除（适配 fine_to_coarse=4）。"""
    lat_min, lat_max, lon_min, lon_max = region
    li = np.where((lat <= lat_max) & (lat >= lat_min))[0]
    lj = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    # 裁剪到能被 4 整除
    h = len(li)
    w = len(lj)
    h4 = (h // 4) * 4
    w4 = (w // 4) * 4
    trim_h = h - h4
    trim_w = w - w4
    li = li[trim_h // 2 : trim_h // 2 + h4]
    lj = lj[trim_w // 2 : trim_w // 2 + w4]
    return li, lj


def _select_subgrid(
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    row: int,
    col: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a square subgrid from the already-cropped Beijing grid."""
    if size <= 0 or size % 4 != 0:
        raise ValueError(f"subgrid size must be a positive multiple of 4, got {size}")
    if row < 0 or col < 0 or row + size > len(lat_idx) or col + size > len(lon_idx):
        raise ValueError(
            f"Subgrid row={row}, col={col}, size={size} exceeds cropped grid "
            f"{len(lat_idx)}x{len(lon_idx)}."
        )
    return lat_idx[row : row + size], lon_idx[col : col + size]


def _load_daily_files(
    nc_dir: Path,
    years: list[int],
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
) -> np.ndarray:
    """加载指定年份全部日值 NC 文件，裁切并堆叠为 [T, H, W] 数组。"""
    frames = []
    for year in sorted(years):
        year_dir = nc_dir / f"CHAP_PM2.5_D1K_{year}_V4"
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Year directory not found: {year_dir}")
        nc_files = sorted(year_dir.glob("CHAP_PM2.5_D1K_*_V4.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NC files found in {year_dir}")
        print(f"  {year}: {len(nc_files)} daily files")
        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file, engine="h5netcdf")
            pm = ds["PM2.5"].values  # [H, W]
            pm = np.where(pm >= 65530, np.nan, pm)  # missing → NaN
            crop = pm[lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1]
            frames.append(crop.astype(np.float32))
            ds.close()
    data = np.stack(frames, axis=0)  # [T, H, W]
    return data


def _fill_missing(data: np.ndarray) -> np.ndarray:
    """简单的时间维线性插值填补 NaN，首尾向外填充。"""
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        return data

    filled = data.copy()
    H, W = data.shape[1], data.shape[2]
    for h in range(H):
        for w in range(W):
            ts = data[:, h, w]
            nan_idx = np.isnan(ts)
            if not nan_idx.any():
                continue
            if nan_idx.all():
                filled[:, h, w] = 0.0
                continue
            ok = ~nan_idx
            filled[nan_idx, h, w] = np.interp(
                np.flatnonzero(nan_idx).astype(float),
                np.flatnonzero(ok).astype(float),
                ts[ok],
            )
    return filled


def _make_windows(data: np.ndarray, window: int, stride: int) -> np.ndarray:
    """[T, H, W] → [N, 1, T, H, W]"""
    n = (data.shape[0] - window) // stride + 1
    windows = np.zeros((n, 1, window, data.shape[1], data.shape[2]), dtype=np.float32)
    for i in range(n):
        start = i * stride
        windows[i, 0] = data[start : start + window]
    return windows


def _split_frames(data: np.ndarray, val_ratio=0.1, test_ratio=0.1) -> dict[str, np.ndarray]:
    """先按时间顺序切分原始日帧（窗口化之前），避免跨集泄漏。"""
    n = data.shape[0]
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    return {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val :],
    }


def parse_args():
    p = argparse.ArgumentParser(description="CHAP PM2.5 NetCDF → NPZ")
    p.add_argument("--nc_dir", default="data/CHAP", help="CHAP NetCDF 根目录")
    p.add_argument("--out_dir", default="data/CHAP", help="输出 NPZ 目录")
    p.add_argument("--region", default="bj", choices=list(REGIONS.keys()), help="裁切区域")
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021])
    p.add_argument("--window", type=int, default=7, help="滑动窗口长度(天)")
    p.add_argument("--stride", type=int, default=1, help="滑动步长(天)")
    p.add_argument(
        "--subgrid_row",
        type=int,
        default=77,
        help="北京 168x220 裁切网格内子区域的起始行（默认选定的有效背景块）",
    )
    p.add_argument(
        "--subgrid_col",
        type=int,
        default=52,
        help="北京 168x220 裁切网格内子区域的起始列（默认选定的有效背景块）",
    )
    p.add_argument("--subgrid_size", type=int, default=32, help="正方形子区域边长")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    return p.parse_args()


def main():
    if not HAS_XARRAY:
        raise ImportError("需要 xarray + h5netcdf: pip install xarray h5netcdf")

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    nc_dir = root / args.nc_dir
    out_dir = root / args.out_dir / "beijing"
    out_dir.mkdir(parents=True, exist_ok=True)

    region = REGIONS[args.region]

    # 查找第一个文件获取 lat/lon 和裁切索引
    sample_files = sorted((nc_dir / "CHAP_PM2.5_D1K_2019_V4").glob("*.nc"))
    if not sample_files:
        sample_files = sorted((nc_dir / f"CHAP_PM2.5_D1K_{args.years[0]}_V4").glob("*.nc"))
    ds = xr.open_dataset(str(sample_files[0]), engine="h5netcdf")
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()

    full_lat_idx, full_lon_idx = _find_indices(lat, lon, region)
    lat_idx, lon_idx = _select_subgrid(
        full_lat_idx,
        full_lon_idx,
        args.subgrid_row,
        args.subgrid_col,
        args.subgrid_size,
    )
    h, w = len(lat_idx), len(lon_idx)
    lat_vals = lat[lat_idx]
    lon_vals = lon[lon_idx]

    print("=" * 60)
    print("  CHAP → NPZ 预处理")
    print(f"  区域: {args.region} ({REGIONS[args.region]})")
    print(f"  北京完整裁切: {len(full_lat_idx)}×{len(full_lon_idx)} (H×W)")
    print(f"  选定子区域: row={args.subgrid_row}, col={args.subgrid_col}, size={args.subgrid_size}")
    print(f"  输出网格: {h}×{w} (H×W)")
    print(f"  lat: [{lat_vals[0]:.4f}, {lat_vals[-1]:.4f}]")
    print(f"  lon: [{lon_vals[0]:.4f}, {lon_vals[-1]:.4f}]")
    print(f"  窗口: T={args.window}, stride={args.stride}")
    print(f"  年份: {args.years}")
    print("=" * 60)
    print()

    # 加载全部日值数据
    print("[1/4] 加载每日 NC 文件...")
    data = _load_daily_files(nc_dir, args.years, lat_idx, lon_idx)
    print(f"  总帧数: {data.shape[0]}, 网格: {data.shape[1]}×{data.shape[2]}")
    nan_count = np.isnan(data).sum()
    print(f"  NaN 格点数: {nan_count:,} ({nan_count / data.size * 100:.2f}%)")

    # 填补缺失
    if nan_count > 0:
        print("[2/4] 线性插值填补 NaN...")
        data = _fill_missing(data)
        print(f"  填补后 NaN: {np.isnan(data).sum():,}")
    else:
        print("[2/4] 无缺失值，跳过填补")

    # 先切分原始日帧（避免窗口重叠导致的跨集泄漏）
    print("[3/4] 按时序切分原始日帧 (train/val/test)...")
    frame_splits = _split_frames(data, args.val_ratio, args.test_ratio)
    for name, frames in frame_splits.items():
        print(f"  {name}: {frames.shape[0]} days")

    # 各 split 独立窗口化
    print("[4/4] 各自独立构造滑动窗口并保存...")
    for name, frames in frame_splits.items():
        wins = _make_windows(frames, args.window, args.stride)
        path = out_dir / f"chap_beijing_{name}.npz"
        np.savez_compressed(path, x_f_gt=wins)
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {wins.shape[0]} windows × ({wins.shape[2]}×{wins.shape[3]}×{wins.shape[4]}) → {path} ({size_mb:.1f} MB)")

    # 保存元信息
    meta_path = out_dir / "meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"region: {args.region}\n")
        f.write(f"source_grid: {len(full_lat_idx)}×{len(full_lon_idx)}\n")
        f.write(f"subgrid_row: {args.subgrid_row}\n")
        f.write(f"subgrid_col: {args.subgrid_col}\n")
        f.write(f"subgrid_size: {args.subgrid_size}\n")
        f.write(f"lat_range: [{lat_vals[0]:.4f}, {lat_vals[-1]:.4f}]\n")
        f.write(f"lon_range: [{lon_vals[0]:.4f}, {lon_vals[-1]:.4f}]\n")
        f.write(f"grid: {h}×{w}\n")
        f.write(f"window: {args.window}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"years: {args.years}\n")
        f.write(f"train_days: {frame_splits['train'].shape[0]}\n")
        f.write(f"val_days: {frame_splits['val'].shape[0]}\n")
        f.write(f"test_days: {frame_splits['test'].shape[0]}\n")
    print(f"  → {meta_path}")
    print()
    print("完成！")


if __name__ == "__main__":
    main()
