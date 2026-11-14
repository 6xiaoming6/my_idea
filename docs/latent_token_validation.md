# Independent latent token acceptance (2026-11-14)

Preflight evidence only; not scientific training outcomes.

- 25 CPU tests passed, including original release/performance regression, grid experiments, token contracts, full CLI continuation, and report rendering.
- After adding no-grad token diagnostics, all six token contract tests passed again; updated report rendering also passed.
- Tested K=97 and token-slot permutation; no 2D grid assignment. Common old parameters, prefix slot bank, mapping templates, data split and masks verified.
- GPU1 batch32 AMP: all ten token configurations passed native routing plus last-F TA/ST backward. Exact full state continuation for LT01/LT03/LT10.
- Additional LT03 K=256, FFFF, TA/ST in all four rounds passed forward/backward. Peak allocated 21.422329425811768 GiB, reserved 21.791015625 GiB.
- No batch reduction, no GPU0 training.

| Variant | Total params | Trainable params | Native step seconds | Last-F TA/ST allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| LT01 | 376602 | 361691 | 0.8988 | 14.7050 |
| LT02 | 372506 | 357595 | 0.4620 | 12.7499 |
| LT03 | 384794 | 369883 | 0.5809 | 18.6388 |
| LT04 | 417562 | 402651 | 0.5504 | 14.6994 |
| LT05 | 376602 | 361691 | 0.5051 | 14.7076 |
| LT06 | 376602 | 340571 | 0.4975 | 14.6841 |
| LT07 | 376602 | 359258 | 0.4473 | 12.7250 |
| LT08 | 376602 | 338138 | 0.4435 | 12.7025 |
| LT09 | 376602 | 316958 | 0.4995 | 11.9464 |
| LT10 | 417562 | 402651 | 0.5061 | 14.7008 |

Times are isolated preflight steps, including diagnostics/startup effects and route differences, not steady-state or full-training guarantees. All formal results remain pending. Grid-suite L01 was interrupted after 80 committed epochs on user clarification and is excluded from the token suite.
