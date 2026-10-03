# Objective Checkpoint Report

Run: `/runs/h3/ref2va-lora-long-1500`

| checkpoint | step | val video loss | improvement vs step 0 | monotonic | remote checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| ckpt_250 | 250 | 0.261144 | 58.7% |  | `/runs/h3/ref2va-lora-long-1500/checkpoint-250.pt` |
| ckpt_500 | 500 | 0.251213 | 60.3% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-500.pt` |
| ckpt_750 | 750 | 0.248858 | 60.7% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-750.pt` |
| ckpt_1000 | 1000 | 0.240677 | 62.0% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-1000.pt` |
| ckpt_1250 | 1250 | 0.235390 | 62.8% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-1250.pt` |
| ckpt_1500 | 1500 | 0.241075 | 61.9% | false | `/runs/h3/ref2va-lora-long-1500/checkpoint-1500.pt` |

This is not the final avatar quality score. It proves the training objective is moving in the right direction; generated video review is the next quality gate.
