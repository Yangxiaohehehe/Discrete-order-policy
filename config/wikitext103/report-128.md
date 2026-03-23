# WikiText103 128-Block Experiment Report

Date: 2026-03-23

## Setup

- Dataset: `wikitext103`
- W&B project: `ao-gpt-experiments-128`
- `block_size = 128`
- `batch_size = 64`
- `gradient_accumulation_steps = 4`
- `max_iters = 7000`
- `eval_interval = 500`
- `eval_iters = 200`
- GPUs used during scheduling: both 4090s were used when available, while avoiding interference with existing shared-GPU jobs
- Output root: `nano/nanoGPT/results/wikitext103-wandb-128-20260323/`

## Summary

| Run | Model / train mode | `permute_data` | Best val loss | Best step | Final val `generalization_ar_loss` | W&B |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `ar` | `aogpt / AR` | `False` | `4.1352` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/i1fg9pyl) |
| `gpt` | `gpt / AR` | `False` | `4.3821` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/19xgb9tf) |
| `ar-permute` | `aogpt / AR` | `True` | `4.8873` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/p7mnzr74) |
| `random-permute` | `aogpt / Random` | `True` | `5.8096` | `7000` | `5.8138` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/enrax2s6) |
| `random` | `aogpt / Random` | `False` | `6.3188` | `7000` | `6.3701` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/vpmpwfvz) |

## Main findings

1. Under `block_size = 128`, the best run is still `ar`, with val loss `4.1352`.
2. `gpt` remains second-best at `4.3821`, about `0.2469` worse than `ar`.
3. As in the 256-block setting, permutation hurts AR training noticeably: `ar-permute` is worse than `ar` by about `0.7521` val loss.
4. The Random family is still much worse than the AR family on the main validation metric.
5. Compared with the previous 256-block run, a notable change appears inside the Random family: at 128 block, `random-permute` (`5.8096`) is clearly better than `random` (`6.3188`).
6. The same pattern also appears on AR generalization for Random training: `random-permute` has `generalization_ar_loss = 5.8138`, better than `random` at `6.3701`.
7. All five runs achieve their best reported validation metric at step `7000`, so this training budget still does not show a clear overfitting turning point.

## Run locations

- `gpt`: `results/wikitext103-wandb-128-20260323/gpt`
- `random`: `results/wikitext103-wandb-128-20260323/random`
- `ar`: `results/wikitext103-wandb-128-20260323/ar`
- `random-permute`: `results/wikitext103-wandb-128-20260323/random-permute`
- `ar-permute`: `results/wikitext103-wandb-128-20260323/ar-permute`

## Logs

- Local logs: `nano/nanoGPT/results/wikitext103-wandb-128-20260323/logs/`
- Each run also has a synced W&B page linked above.
