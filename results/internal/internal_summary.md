| Config | n seeds | Accuracy (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | Time (ms/img) |
|---|---|---|---|---|---|---|
| direct_none | 5 | 96.60 ± 0.78 | 96.91 ± 0.83 | 95.83 ± 1.32 | 95.89 ± 1.20 | 47.0 ± 19.3 |
| direct_uniform | 5 | 97.87 ± 0.50 | 98.13 ± 0.35 | 97.18 ± 0.94 | 97.34 ± 0.68 | 41.4 ± 19.4 |
| direct_targeted | 5 | 97.02 ± 1.17 | 97.59 ± 1.00 | 96.49 ± 2.05 | 96.62 ± 1.82 | 39.3 ± 13.6 |
| center_targeted | 5 | 95.46 ± 1.47 | 95.93 ± 1.39 | 95.16 ± 1.59 | 95.20 ± 1.51 | 43.0 ± 21.7 |
| yolo_targeted | 5 | 97.30 ± 1.05 | 97.45 ± 1.29 | 96.68 ± 1.40 | 96.76 ± 1.33 | 68.7 ± 28.3 |
| direct_targeted_resnet50 | 5 | 97.16 ± 1.23 | 97.90 ± 0.67 | 96.95 ± 1.90 | 97.08 ± 1.48 | 165.7 ± 103.3 |
| direct_targeted_mobilenet_v3_large | 5 | 95.89 ± 2.15 | 96.42 ± 2.13 | 95.14 ± 2.34 | 95.14 ± 2.64 | 27.5 ± 11.3 |
| direct_targeted_efficientnet_b0 | 5 | 95.18 ± 2.98 | 95.96 ± 2.57 | 95.44 ± 2.66 | 95.02 ± 3.21 | 45.3 ± 12.2 |

Diferença pareada em relação a direct_targeted (pontos percentuais, média ± dp entre sementes):

| Config | Δ Accuracy | Δ F1 macro |
|---|---|---|
| direct_none | -0.43 ± 1.29 | -0.73 ± 2.27 |
| direct_uniform | +0.85 ± 1.27 | +0.73 ± 2.01 |
| center_targeted | -1.56 ± 0.59 | -1.42 ± 1.04 |
| yolo_targeted | +0.28 ± 1.19 | +0.14 ± 1.32 |
| direct_targeted_resnet50 | +0.14 ± 0.78 | +0.47 ± 1.12 |
| direct_targeted_mobilenet_v3_large | -1.13 ± 2.54 | -1.48 ± 3.17 |
| direct_targeted_efficientnet_b0 | -1.84 ± 2.77 | -1.60 ± 2.95 |