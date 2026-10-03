# Checkpoint 100 Weight Summary

Checkpoint: `artifacts/training/ref2va-lora-smoke-100/checkpoints/checkpoint-100.pt`
Step: `100`
LoRA tensors: `108`
LoRA parameters: `10,655,744`

## Groups

| group | tensors | params | l2 norm |
| --- | ---: | ---: | ---: |
| time_embedder | 4 | 219,136 | 18.7126 |
| token_refiner | 4 | 401,408 | 29.9279 |
| transformer_blocks | 100 | 10,035,200 | 149.8501 |

## Largest Tensors By L2 Norm

| tensor | shape | l2 | abs max |
| --- | --- | ---: | ---: |
| `transformer_blocks.22.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2789 | 0.2871 |
| `transformer_blocks.42.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2475 | 0.2988 |
| `transformer_blocks.43.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2462 | 0.2930 |
| `transformer_blocks.9.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2418 | 0.2910 |
| `transformer_blocks.27.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2397 | 0.3145 |
| `transformer_blocks.20.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2392 | 0.2793 |
| `transformer_blocks.16.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2319 | 0.3184 |
| `transformer_blocks.34.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2301 | 0.3418 |
| `transformer_blocks.36.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2247 | 0.2754 |
| `transformer_blocks.4.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2241 | 0.2695 |
| `transformer_blocks.38.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2229 | 0.2930 |
| `transformer_blocks.25.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2213 | 0.3066 |
