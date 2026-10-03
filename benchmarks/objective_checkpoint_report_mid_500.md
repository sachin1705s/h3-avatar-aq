# Mid-500 Objective Checkpoint Report

Run: `/runs/h3/ref2va-lora-mid-500`

| step | val video loss | improvement vs step 0 | note |
| ---: | ---: | ---: | --- |
| 0 | 0.632806 | 0.0% | baseline |
| 100 | 0.296290 | 53.2% | improved |
| 200 | 0.263503 | 58.4% | improved |
| 300 | 0.254593 | 59.8% | improved |
| 400 | 0.251537 | 60.3% | best objective checkpoint |
| 500 | 0.255668 | 59.6% | worse than previous |

Best objective checkpoint: **step 400** (`loss_v=0.251537`).

Next gate: visual eval for checkpoint 400 versus 500, because step 500 regressed slightly on validation loss.
