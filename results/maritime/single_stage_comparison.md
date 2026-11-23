Teste marítimo: 117 frames, 107 caixas; limiar 0.25.

| Esquema | Det. P / R | E2E P / R | Top-1 por frame | Falso alarme (quadros sem bandeira) | mAP@0.5 (multi) | mAP@0.5:0.95 (multi) | ms/frame |
|---|---|---|---|---|---|---|---|
| two_stage | 0.732 / 0.944 | 0.667 / 0.860 | 0.869 | 0.10 | nan | nan | 73.5 ± 18.1 (med 72.2) |
| single_e30 | 0.728 / 0.626 | 0.283 / 0.243 | 0.243 | 0.00 | 0.381 | 0.263 | 39.0 ± 4.8 (med 38.4) |
| single_e100 | 0.880 / 0.617 | 0.680 / 0.477 | 0.477 | 0.00 | 0.690 | 0.479 | 38.8 ± 5.1 (med 38.3) |

Diferença pareada do recall ponta a ponta (estágio único − dois estágios), IC bootstrap 95%:

| Estágio único | Diferença | IC 95% | só estágio único acerta | só dois estágios acerta | n |
|---|---|---|---|---|---|
| single_e30 | -0.617 | [-0.710, -0.514] | 2 | 68 | 107 |
| single_e100 | -0.383 | [-0.486, -0.280] | 4 | 45 | 107 |

Varredura de limiar (E2E P / R):

| Esquema | 0.1 | 0.25 | 0.5 | 0.7 |
|---|---|---|---|---|
| two_stage | 0.495 / 0.869 | 0.667 / 0.860 | 0.830 / 0.822 | 0.876 / 0.729 |
| single_e30 | 0.183 / 0.308 | 0.283 / 0.243 | 0.422 / 0.178 | 0.714 / 0.140 |
| single_e100 | 0.455 / 0.514 | 0.680 / 0.477 | 0.796 / 0.402 | 0.841 / 0.346 |

dispositivo=cpu, lote=1, frame ja decodificado, imgsz=512, conf=0.25, 3 passagens alternadas; dois estagios = detector + NMS + recortes + classificacao de todas as caixas; estagio unico = uma passagem do YOLO multiclasse com NMS