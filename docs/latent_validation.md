# Latent suite implementation acceptance (2026-11-14)

These are preflight checks, not training outcomes.

- 17 CPU release, performance, latent, and full CLI continuation tests passed.
- Additional report test covers incomplete/complete ten-run rendering and category weighting.
- GPU1, actual TaxiBJ batch32, AMP: ten variants native routing and final-round forced TA/ST passed; 8 expert calls per sample.
- Full model/optimizer/scheduler/scaler/CPU+CUDA RNG continuation matched bitwise for L01/L04/L10.
- All variants retain identical untrained I37 common parameters. F-grid linear compatibility matched reference forward/loss/gradients/optimizer/RNG on CPU.
- No GPU0 training, no change to effective batch size or fixed 0.5/0.5 losses.

| Variant | Params | Trainable | Native step seconds | TA/ST peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| L01 | 347930 | 333019 | 0.8980 | 8.7978 |
| L02 | 347930 | 333019 | 0.3654 | 8.5156 |
| L03 | 372506 | 357595 | 0.3854 | 9.1710 |
| L04 | 413466 | 398555 | 0.3699 | 9.0112 |
| L05 | 347930 | 333019 | 0.3817 | 8.7923 |
| L06 | 347930 | 311899 | 0.3702 | 8.7576 |
| L07 | 347930 | 330586 | 0.3698 | 8.6362 |
| L08 | 347930 | 309466 | 0.3642 | 8.5989 |
| L09 | 347930 | 288286 | 0.3771 | 6.7375 |
| L10 | 372506 | 357595 | 0.3848 | 9.1734 |

Single preflight timings include diagnostics, startup effects, and differing paths. They are not steady-state speed benchmarks or full-run memory guarantees. Source/data/config/initialization hashes are frozen separately by freeze_latent.py before queue launch.
