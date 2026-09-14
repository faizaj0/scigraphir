# CCMP paired statistics per dataset

better = the gold's rank is lower (better) with CCMP; Δlog10 < 0 = CCMP helps. Per gold document, paired.

### sir4_cs · frame_ccmp · gate on vs off, same weights

stratum of the gold (QUARTET)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 1716 | 833 | 558 | 325 | -0.265 [-0.298, -0.234] | 425 → 424 |
| graph | cross | 383 | 128 | 194 | 61 | -0.012 [-0.082, +0.054] | 73 → 72 |
| graph | same | 1257 | 658 | 347 | 252 | -0.332 [-0.370, -0.295] | 336 → 336 |
| fused | all | 1716 | 514 | 486 | 716 | +0.001 [-0.001, +0.003] | 689 → 684 |
| fused | cross | 383 | 80 | 171 | 132 | +0.008 [+0.004, +0.012] | 126 → 124 |
| fused | same | 1257 | 406 | 294 | 557 | -0.001 [-0.004, +0.001] | 532 → 527 |

### sir4_biology · frame_ccmp · gate on vs off, same weights

stratum of the gold (QUARTET)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 1114 | 759 | 271 | 84 | -0.771 [-0.833, -0.711] | 212 → 374 |
| graph | cross | 200 | 109 | 80 | 11 | -0.392 [-0.544, -0.246] | 36 → 51 |
| graph | same | 891 | 640 | 181 | 70 | -0.868 [-0.939, -0.799] | 170 → 315 |
| fused | all | 1114 | 338 | 252 | 524 | -0.003 [-0.009, +0.002] | 677 → 680 |
| fused | cross | 200 | 53 | 88 | 59 | +0.020 [+0.005, +0.034] | 87 → 86 |
| fused | same | 891 | 276 | 157 | 458 | -0.008 [-0.015, -0.002] | 578 → 582 |

### sir4_physics · frame_ccmp · gate on vs off, same weights

stratum of the gold (QUARTET)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 1016 | 716 | 232 | 68 | -0.782 [-0.847, -0.720] | 156 → 348 |
| graph | cross | 240 | 137 | 92 | 11 | -0.337 [-0.471, -0.211] | 34 → 58 |
| graph | same | 731 | 556 | 121 | 54 | -0.966 [-1.037, -0.895] | 112 → 278 |
| fused | all | 1016 | 307 | 230 | 479 | +0.004 [-0.002, +0.010] | 630 → 625 |
| fused | cross | 240 | 68 | 96 | 76 | +0.019 [+0.003, +0.034] | 112 → 107 |
| fused | same | 731 | 230 | 114 | 387 | -0.003 [-0.009, +0.003] | 496 → 496 |

### sir4_matsci · frame_ccmp · gate on vs off, same weights

stratum of the gold (QUARTET)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 983 | 398 | 300 | 285 | -0.115 [-0.146, -0.085] | 349 → 347 |
| graph | cross | 256 | 75 | 109 | 72 | +0.040 [-0.016, +0.097] | 76 → 78 |
| graph | same | 711 | 317 | 185 | 209 | -0.171 [-0.207, -0.140] | 267 → 265 |
| fused | all | 983 | 327 | 335 | 321 | +0.002 [-0.007, +0.010] | 428 → 415 |
| fused | cross | 256 | 74 | 115 | 67 | +0.021 [+0.007, +0.035] | 94 → 91 |
| fused | same | 711 | 248 | 214 | 249 | -0.005 [-0.015, +0.004] | 328 → 318 |

### tomato · frame_ccmp · gate on vs off, same weights

stratum of the query (hops file)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 409 | 118 | 179 | 112 | +0.029 [+0.014, +0.042] | 141 → 139 |
| graph | cross | 289 | 83 | 137 | 69 | +0.038 [+0.022, +0.054] | 84 → 83 |
| graph | same | 120 | 35 | 42 | 43 | +0.005 [-0.018, +0.029] | 57 → 56 |
| fused | all | 409 | 67 | 112 | 230 | +0.002 [-0.000, +0.005] | 186 → 186 |
| fused | cross | 289 | 48 | 93 | 148 | +0.003 [+0.000, +0.006] | 116 → 116 |
| fused | same | 120 | 19 | 19 | 82 | -0.001 [-0.003, +0.002] | 70 → 70 |

### mir · frame_ccmp · gate on vs off, same weights

stratum of the query (hops file)

| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |
|---|---|---|---|---|---|---|---|
| graph | all | 235 | 98 | 80 | 57 | +0.013 [-0.020, +0.047] | 65 → 65 |
| graph | same | 235 | 98 | 80 | 57 | +0.013 [-0.020, +0.047] | 65 → 65 |
| fused | all | 235 | 50 | 33 | 152 | -0.002 [-0.005, +0.001] | 113 → 113 |
| fused | same | 235 | 50 | 33 | 152 | -0.002 [-0.005, +0.001] | 113 → 113 |
