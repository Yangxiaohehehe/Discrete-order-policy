# WikiText103 Permute-Seed Sweep Report (Block 128)

Date: 2026-03-23

## Setup

- Dataset: `wikitext103`
- Focused configs: `ar_permute`, `random_permute`
- Shared fixed settings: `block_size=128`, `batch_size=64`, `gradient_accumulation_steps=4`, `max_iters=7000`
- W&B project: `ao-gpt-experiments-128`
- Newly run seeds in this sweep: `1`, `7`, `123`
- Reference baseline from previous 128-block run: `permute_seed=42`
- New output root: `nano/nanoGPT/results/wikitext103-wandb-128-permute-seeds-20260323/`

## Summary Table

| Config | `permute_seed` | Best val loss | Best step | Final val `generalization_ar_loss` | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| `ar_permute` | `1` | `4.8978` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/gtka6tud) |
| `ar_permute` | `7` | `4.8428` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/rv8ombxt) |
| `ar_permute` | `42` | `4.8873` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/p7mnzr74) |
| `ar_permute` | `123` | `4.8453` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/pcd6l39h) |
| `random_permute` | `1` | `5.7552` | `7000` | `5.7650` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/hb379yl9) |
| `random_permute` | `7` | `5.5655` | `7000` | `5.5672` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/1ymolenw) |
| `random_permute` | `42` | `5.8096` | `7000` | `5.8138` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/enrax2s6) |
| `random_permute` | `123` | `5.5776` | `7000` | `5.5737` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-128/runs/36ev9ne2) |

## Main Observations

1. `permute_seed` does matter at `block_size=128`, especially for `random_permute`.
2. For `ar_permute`, the seed sensitivity is present but fairly small: best val loss ranges from `4.8428` to `4.8978`, a spread of about `0.0550`.
3. For `random_permute`, the seed sensitivity is much larger: best val loss ranges from `5.5655` to `5.8096`, a spread of about `0.2441`.
4. Among the four observed seeds, `permute_seed=7` is the best for both configs.
5. `permute_seed=123` is very close to `7` for `ar_permute`, and clearly better than `1` and `42` for `random_permute`.
6. The previous baseline `permute_seed=42` is not the best seed for either config under `block_size=128`.
7. All runs again achieve their best reported metric at step `7000`, so within this budget the training curve is still improving through the end.

## Interpretation Notes

- This sweep supports the idea that the fixed permutation itself is acting like a meaningful choice of coordinate system, not just a harmless implementation detail.
- The effect is much stronger in `random_permute` than in `ar_permute`, which suggests Random-mode training is more sensitive to which fixed permutation is chosen.
- If you want a stronger conclusion, the next natural step is to run a few more seeds and report mean/std for each config.

## Run Locations

- `ar_permute seed 1`: `results/wikitext103-wandb-128-permute-seeds-20260323/ar-permute-seed1`
- `ar_permute seed 7`: `results/wikitext103-wandb-128-permute-seeds-20260323/ar-permute-seed7`
- `ar_permute seed 123`: `results/wikitext103-wandb-128-permute-seeds-20260323/ar-permute-seed123`
- `random_permute seed 1`: `results/wikitext103-wandb-128-permute-seeds-20260323/random-permute-seed1`
- `random_permute seed 7`: `results/wikitext103-wandb-128-permute-seeds-20260323/random-permute-seed7`
- `random_permute seed 123`: `results/wikitext103-wandb-128-permute-seeds-20260323/random-permute-seed123`

## Logs

- New sweep logs: `nano/nanoGPT/results/wikitext103-wandb-128-permute-seeds-20260323/logs/`
- Baseline 128 logs used for seed `42`: `nano/nanoGPT/results/wikitext103-wandb-128-20260323/logs/`
