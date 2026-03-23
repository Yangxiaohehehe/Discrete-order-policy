# WikiText103 Experiment Results

Date: 2026-03-23

## Run setup

- Dataset: `wikitext103`
- Train steps: `7000`
- Eval interval: `500`
- Eval iters: `200`
- Batch size: `32`
- Block size: `256`
- Gradient accumulation: `4`
- GPUs used: `2 x RTX 4090`
- Execution mode: two single-GPU jobs in parallel
- `wandb_log` was overridden to `False` for these local runs
- Output root: `nano/nanoGPT/results/wikitext103-20260323/`

## Summary table

| Run | Source config | Model / train mode | `permute_data` | Best val loss | Best step | Final val loss | Final val `generalization_ar_loss` | Output dir |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `ar` | `config/WikiText103/ar.py` | `aogpt / AR` | `False` | `4.0989` | `7000` | `4.0989` | `-` | `results/wikitext103-20260323/ar` |
| `gpt` | `config/WikiText103/gpt.py` | `gpt / AR` | `False` | `4.3686` | `6500` | `4.3739` | `-` | `results/wikitext103-20260323/gpt` |
| `ar-permute` | `config/WikiText103/ar_permute.py` | `aogpt / AR` | `True` | `4.8665` | `7000` | `4.8665` | `-` | `results/wikitext103-20260323/ar-permute` |
| `random` | `config/WikiText103/random.py` | `aogpt / Random` | `False` | `6.7903` | `7000` | `6.7903` | `6.7781` | `results/wikitext103-20260323/random` |
| `random-permute` | `config/WikiText103/random_permute.py` | `aogpt / Random` | `True` | `6.7906` | `7000` | `6.7906` | `6.7906` | `results/wikitext103-20260323/random-permute` |

## Main observations

1. `ar` is the best run on WikiText103 validation loss: `4.0989`.
2. `gpt` is worse than `ar` by about `0.2697` val loss, but still clearly better than `ar-permute`.
3. Adding token permutation hurts AR training a lot: `ar-permute` is worse than `ar` by about `0.7676` val loss.
4. The two Random-order runs are much worse on the main validation metric than the AR-family runs.
5. For Random training, non-permuted data generalizes slightly better to AR evaluation: `6.7781` vs `6.7906`.
6. Most runs still reach their best metric at step `7000`, so this budget does not yet show obvious overfitting for those settings. `gpt` is the exception: its best val loss appears at step `6500` and is slightly worse at the final checkpoint.

## Logs and checkpoints

- Logs: `nano/nanoGPT/results/wikitext103-20260323/logs/`
- Checkpoints: one `ckpt.pt` under each run directory in `nano/nanoGPT/results/wikitext103-20260323/`

## Notes

- I overrode `out_dir` per run to avoid collisions. In the original configs, `ar.py` and `ar_permute.py` both point to `out-wikitext103-ar`.
- During execution, Random-mode training exposed a bug in `AOGPT.py`: `torchf.stack(...)` in `sample_random_orders()` caused the Random configs to crash. I fixed it to `torch.stack(...)` and added `tests/test_aogpt_random_orders.py` as a regression test before rerunning the affected experiments.
