Teste marítimo: 117 frames, 107 caixas. Detector fixo; limiar 0.25.

| Modelo | Params (M) | Oráculo acc | Oráculo F1 | Det. P / R | E2E P / R | Top-1 por frame | Classif. ms/recorte | Pipeline ms/frame |
|---|---|---|---|---|---|---|---|---|
| resnet18 | 11.19 | 0.925 | 0.924 | 0.732 / 0.944 | 0.667 / 0.860 | 0.869 | 30.2 ± 1.9 (med 30.1) | 72.6 ± 18.4 (med 71.8) |
| mobilenet_v3_large | 4.23 | 0.897 | 0.892 | 0.732 / 0.944 | 0.638 / 0.822 | 0.832 | 19.2 ± 2.6 (med 20.2) | 60.2 ± 12.7 (med 59.9) |

Diferenças pareadas (MobileNetV3-L − ResNet18), IC bootstrap 95% e discordâncias:

| Medida | Diferença | IC 95% | só MobileNet acerta | só ResNet acerta | n |
|---|---|---|---|---|---|
| oracle | -0.028 | [-0.075, +0.019] | 2 | 5 | 107 |
| e2e | -0.037 | [-0.093, +0.009] | 2 | 6 | 107 |
| frame_top1 | -0.037 | [-0.093, +0.009] | 2 | 6 | 107 |

dispositivo=cpu, lote=1, frame ja decodificado, imgsz=512, conf=0.25; 3 passagens alternadas; 'pipeline' = detector + NMS + recortes + classificacao de todas as caixas; 'clf_only' = pre-processamento + classificador por recorte anotado