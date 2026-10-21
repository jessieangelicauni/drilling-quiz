# Evaluation protocol (draft, not yet frozen)

Added 2026-10-21 with `src/eval_harness.py`. It's a draft until the items below are agreed and the file is tagged in git. Tag it, for example `eval-protocol-v1`, before the test split is run for the paper. After that, changing a metric, threshold or the split salt means a new version of this file, recorded as such.

## Split

- Unit: one generated item, identified by (skill, stem). The curated items of the removed template generator (`1_validation_results.json`, `log_likelihood_results.json`) are the historical record only.
- Split: `split_of()` in `src/eval_harness.py`, salt `ko-drill-eval-v1`, 30% test.
- Rule: prompts, the codebook-building prompt and thresholds are tuned on **dev** items only. Paper numbers come from **test** only. A test item that fails is reported. It isn't edited and re-run.
- The misconception codebook (`data/misconception_codebook.json`) is built once and frozen before the test run; rebuilding it is a new version of this protocol.

## Metrics (all reported with Wilson 95% intervals)

| Metric | Source | Note |
|---|---|---|
| Item acceptance (untargeted / targeted) | `validate_item_foundry.py` | Share of requests that pass every check in `item_foundry.verify` within two revisions, plus rejection reasons |
| Codebook coverage | `validate_item_foundry.py` | Distinct misconception codes the accepted items cover, per skill |
| Solver key pick, rotated | `validate_item_foundry.py` | Qwen3-4B letter pick under every rotation; reported, not a gate (it judges, PRD §9.16) |
| Key outscores each distractor | gate inside `item_foundry.verify` | Same model as the gate, so it is not independent evidence; its calibration is the curated-pair result (51/52) |
| Misconception recovery (simulation) | `simulate_misconceptions.py` | Top-1 accuracy after 5/10/20/40 wrong answers, stable and remediated, CI over 20 replications |
| Human / independent rating | to be added | At least 2 raters, Cohen's kappa: key uniqueness, distractor plausibility, and agreement of each distractor's code with the rater's own codebook label; the only metrics independent of the gates |

## Runs

- At least 5 seeds per GPU script. Pass all result files to `eval_harness.py`.
- Each result JSON should record the model name, HF revision, library versions and GPU. Not yet done in the GPU scripts.
