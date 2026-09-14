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
