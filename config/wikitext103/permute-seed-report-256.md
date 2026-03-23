# WikiText103 Permute-Seed Sweep Report (Block 256)

Date: 2026-03-23

## Setup

- Dataset: `wikitext103`
- Focused configs: `ar_permute`, `random_permute`
- Shared fixed settings for this sweep: `block_size=256`, `batch_size=32`, `gradient_accumulation_steps=4`, `max_iters=7000`
- W&B project: `ao-gpt-experiments-256`
- Newly run seeds in this sweep: `1`, `7`, `123`
- Reference baseline from previous 256-block run: `permute_seed=42`
- New output root: `nano/nanoGPT/results/wikitext103-wandb-256-permute-seeds-20260323/`

## Summary Table

| Config | `permute_seed` | Best val loss | Best step | Final val `generalization_ar_loss` | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| `ar_permute` | `1` | `4.8581` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/eloa556w) |
| `ar_permute` | `7` | `4.8851` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/y5wr9d2r) |
| `ar_permute` | `42` | `4.8595` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/r3mimhj3) |
| `ar_permute` | `123` | `4.8451` | `7000` | `-` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/iryyf037) |
| `random_permute` | `1` | `6.7947` | `7000` | `6.7948` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/oimp37qy) |
| `random_permute` | `7` | `6.8007` | `7000` | `6.8008` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/3neufmal) |
| `random_permute` | `42` | `6.8004` | `7000` | `6.8002` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/pcrwvlb3) |
| `random_permute` | `123` | `6.7977` | `7000` | `6.7977` | [run](https://wandb.ai/chenheyang15/ao-gpt-experiments-256/runs/ukn8jtop) |

## Main Observations

1. Under `block_size=256`, `permute_seed` still matters for `ar_permute`, but the sensitivity is moderate: val loss ranges from `4.8451` to `4.8851`, a spread of about `0.0400`.
2. For `ar_permute`, the best seed among the observed four is `123`, followed very closely by `1` and `42`.
3. For `random_permute`, seed sensitivity at 256 is much smaller than what we saw at 128: val loss ranges only from `6.7947` to `6.8007`, a spread of about `0.0060`.
4. For `random_permute`, all four seeds are effectively very close; `seed=1` is the best in this sweep, but only slightly better than `123`, `42`, and `7`.
5. Compared with the 128-block sweep, the seed effect is clearly weaker at 256, especially for `random_permute`.
6. As in earlier runs, all reported best metrics are still reached at step `7000`.

## Interpretation Notes

- This result is consistent with the idea that larger context length makes the model less sensitive to the exact fixed permutation used in `random_permute`.
- At `block_size=256`, the permutation seed still influences `ar_permute`, but the effect is small enough that the main qualitative picture is stable across seeds.
- The much larger seed spread at `block_size=128` now looks like a genuine short-context sensitivity effect rather than simple training noise.

## Run Locations

- `ar_permute seed 1`: `results/wikitext103-wandb-256-permute-seeds-20260323/ar-permute-seed1`
- `ar_permute seed 7`: `results/wikitext103-wandb-256-permute-seeds-20260323/ar-permute-seed7`
- `ar_permute seed 123`: `results/wikitext103-wandb-256-permute-seeds-20260323/ar-permute-seed123`
- `random_permute seed 1`: `results/wikitext103-wandb-256-permute-seeds-20260323/random-permute-seed1`
- `random_permute seed 7`: `results/wikitext103-wandb-256-permute-seeds-20260323/random-permute-seed7`
- `random_permute seed 123`: `results/wikitext103-wandb-256-permute-seeds-20260323/random-permute-seed123`

## Logs

- New sweep logs: `nano/nanoGPT/results/wikitext103-wandb-256-permute-seeds-20260323/logs/`
- Baseline 256 logs used for seed `42`: `nano/nanoGPT/results/wikitext103-wandb-20260323/logs/`
