# Recall by flag size (sqrt of box area in pixels, 512-px long side)

## A. Test boxes as photographed (scale 1.0)

| size (px) | boxes | detected | detection recall (95% CI) | end-to-end recall |
|---|---|---|---|---|
| <16 | 0 | - | - | - |
| 16-32 | 1 | 1 | 1.000 (0.21-1.00) | 1.000 |
| 32-64 | 15 | 14 | 0.933 (0.70-0.99) | 0.800 |
| 64-96 | 18 | 16 | 0.889 (0.67-0.97) | 0.778 |
| >=96 | 73 | 70 | 0.959 (0.89-0.99) | 0.890 |

## B. Simulated distance: all scales pooled by effective size

| effective size (px) | boxes | detected | detection recall (95% CI) | end-to-end recall |
|---|---|---|---|---|
| <16 | 54 | 3 | 0.056 (0.02-0.15) | 0.019 |
| 16-32 | 149 | 71 | 0.477 (0.40-0.56) | 0.450 |
| 32-64 | 180 | 150 | 0.833 (0.77-0.88) | 0.783 |
| 64-96 | 62 | 56 | 0.903 (0.80-0.95) | 0.839 |
| >=96 | 90 | 86 | 0.956 (0.89-0.98) | 0.900 |

## Size distribution of annotated boxes (512-px scale)

- train: n=3, min=309.8, median=335.2, <16: 0, 16-32: 0, 32-64: 0, 64-96: 0, >=96: 3
- val: n=1, min=137.1, median=137.1, <16: 0, 16-32: 0, 32-64: 0, 64-96: 0, >=96: 1
- test: n=107, min=28.0, median=122.3, <16: 0, 16-32: 1, 32-64: 15, 64-96: 18, >=96: 73
