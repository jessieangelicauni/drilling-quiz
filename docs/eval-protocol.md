# Evaluation protocol (draft, not yet frozen)

Added 2026-10-21 with `src/eval_harness.py`. It's a draft until the items below are agreed and the file is tagged in git. Tag it, for example `eval-protocol-v1`, before the test split is run for the paper. After that, changing a metric, threshold or the split salt means a new version of this file, recorded as such.

## Split

- Unit: one curated item, identified by (skill, stem).
- Split: `split_of()` in `src/eval_harness.py`, salt `ko-drill-eval-v1`, 30% test.
- Rule: prompts, curated pairs, filler sets and thresholds are tuned on **dev** only. Paper numbers come from **test** only. A test item that fails is reported. It isn't edited and re-run.

## Metrics (all reported per unique item with Wilson 95% intervals)

| Metric | Source | Note |
|---|---|---|
| Solver key acceptance (check c) | `validate_novelty1.py` | Report the letter pick and the debiased picks (`pick_by_logprob`, `pick_best_rotated`) side by side, plus accuracy by key position |
| Correct form outscores distractor | `validate_log_likelihood.py` | Same model family as the gate, so this is not independent evidence of item quality |
| kiwipiepy parse score | `validate_lm_plausibility.py` | Non-LLM |
| Human / independent rating | to be added | At least 2 raters, Cohen's kappa; the only metric independent of the gates |

## Runs

- At least 5 seeds per GPU script. Pass all result files to `eval_harness.py`.
- Each result JSON should record the model name, HF revision, library versions and GPU. Not yet done in the GPU scripts.
