Teste marítimo: 117 frames, 107 caixas anotadas, 10 frames sem bandeira.

**Detector (caixas reais, classe única)**

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| val | 0.995 | 0.695 | 1.000 | 0.999 |
| test | 0.940 | 0.616 | 0.897 | 0.907 |

**Classificador A** (limiar operacional conf=0.25)

| Métrica | Valor |
|---|---|
| Detecção: precision / recall / F1 (IoU≥0.5) | 0.732 / 0.944 / 0.824 |
| Acurácia de país nas caixas casadas | 0.891 |
| Ponta-a-ponta: precision / recall | 0.652 / 0.841 |
| Falso alarme em frames sem bandeira (fração) / FP por frame negativo | 0.100 / 0.100 |
| Oráculo (caixa anotada→classificador): acurácia / F1 macro (n=107) | 0.897 / 0.899 |

Acurácia top-1 por frame (n=107 frames com bandeira) — frame inteiro: 0.299 | recorte central 80%: 0.449 | caixa do YOLO de maior confiança: 0.850

Varredura de limiar:

| conf | det P | det R | e2e P | e2e R | falso alarme (frames neg.) |
|---|---|---|---|---|---|
| 0.1 | 0.543 | 0.953 | 0.484 | 0.850 | 0.200 |
| 0.25 | 0.732 | 0.944 | 0.652 | 0.841 | 0.100 |
| 0.5 | 0.915 | 0.907 | 0.821 | 0.813 | 0.000 |
| 0.7 | 0.955 | 0.794 | 0.865 | 0.720 | 0.000 |

Subgrupos:

| Subgrupo | n caixas | det. recall | e2e recall | acurácia oráculo |
|---|---|---|---|---|
| size=small | 0 | nan | nan | nan |
| size=medium | 27 | 0.889 | 0.741 | 0.852 |
| size=large | 80 | 0.963 | 0.875 | 0.912 |
| blurred=1 | 20 | 0.900 | 0.750 | 0.850 |
| occluded=1 | 19 | 0.842 | 0.632 | 0.842 |
| deformed=1 | 76 | 0.921 | 0.789 | 0.868 |
| clean (none of blurred/occluded/deformed) | 25 | 1.000 | 0.960 | 0.960 |

**Classificador B** (limiar operacional conf=0.25)

| Métrica | Valor |
|---|---|
| Detecção: precision / recall / F1 (IoU≥0.5) | 0.732 / 0.944 / 0.824 |
| Acurácia de país nas caixas casadas | 0.911 |
| Ponta-a-ponta: precision / recall | 0.667 / 0.860 |
| Falso alarme em frames sem bandeira (fração) / FP por frame negativo | 0.100 / 0.100 |
| Oráculo (caixa anotada→classificador): acurácia / F1 macro (n=107) | 0.925 / 0.924 |

Acurácia top-1 por frame (n=107 frames com bandeira) — frame inteiro: 0.449 | recorte central 80%: 0.477 | caixa do YOLO de maior confiança: 0.869

Varredura de limiar:

| conf | det P | det R | e2e P | e2e R | falso alarme (frames neg.) |
|---|---|---|---|---|---|
| 0.1 | 0.543 | 0.953 | 0.495 | 0.869 | 0.200 |
| 0.25 | 0.732 | 0.944 | 0.667 | 0.860 | 0.100 |
| 0.5 | 0.915 | 0.907 | 0.830 | 0.822 | 0.000 |
| 0.7 | 0.955 | 0.794 | 0.876 | 0.729 | 0.000 |

Subgrupos:

| Subgrupo | n caixas | det. recall | e2e recall | acurácia oráculo |
|---|---|---|---|---|
| size=small | 0 | nan | nan | nan |
| size=medium | 27 | 0.889 | 0.741 | 0.852 |
| size=large | 80 | 0.963 | 0.900 | 0.950 |
| blurred=1 | 20 | 0.900 | 0.800 | 0.900 |
| occluded=1 | 19 | 0.842 | 0.684 | 0.842 |
| deformed=1 | 76 | 0.921 | 0.816 | 0.908 |
| clean (none of blurred/occluded/deformed) | 25 | 1.000 | 0.960 | 0.960 |

**Tempo por frame** (dispositivo=cpu, lote=1 frame, frame já decodificado (exclui leitura de disco), imgsz=512, conf=0.25; inclui NMS, recortes, pré-processamento e classificação de todas as caixas detectadas): detector 38.5 ± 4.9 ms; detector+classificador 75.1 ± 18.6 ms