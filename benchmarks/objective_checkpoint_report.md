# Objective Checkpoint Report

Run: `/runs/h3/ref2va-lora-smoke-100`

| checkpoint | step | val video loss | improvement vs step 0 | monotonic | remote checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| ckpt_020 | 20 | 0.598621 | 5.4% |  | `/runs/h3/ref2va-lora-smoke-100/checkpoint-20.pt` |
| ckpt_040 | 40 | 0.498908 | 21.2% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-40.pt` |
| ckpt_060 | 60 | 0.415435 | 34.4% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-60.pt` |
| ckpt_080 | 80 | 0.323771 | 48.8% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-80.pt` |
| ckpt_100 | 100 | 0.293227 | 53.7% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt` |

This is not the final avatar quality score. It proves the training objective is moving in the right direction; generated video review is the next quality gate.
