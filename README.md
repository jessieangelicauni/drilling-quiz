# KO-DRILL — IEEE paper, figures, and simulation

**Mulai dari sini:** [`docs/PRD.md`](docs/PRD.md) — semua alasan desain (kenapa Parallel Elo bukan DKT, kenapa EXAONE/Qwen3/KLUE-RoBERTa, tiga novelty riset, apa yang sengaja TIDAK dikerjakan) jadi satu acuan.

**Literature review:** [`docs/literature-review-15.md`](docs/literature-review-15.md) — 13 curated sources (2024–2026, IEEE/Springer/Elsevier/Wiley/Taylor & Francis/SAGE only, no arXiv/MDPI) with DOI links and explicit research-gap mapping.

Personalised Korean drill quizzes: a psychometrically grounded LLM pipeline.

## Output

- `paper/KO-DRILL_IEEE.docx` — the paper, IEEE two-column format
- `paper/KO-DRILL_IEEE.pdf` — rendered for checking (LibreOffice)

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/simulate_trackers.py   # writes figures/tracker_sim.npz
.venv/bin/python src/make_figures.py        # writes figures/fig1..fig6
python3 src/build_docx.py                   # writes paper/KO-DRILL_IEEE.docx
```

## Implementation (beyond the paper)

Besides the paper/figures pipeline, this repo now has the start of the
actual system, built directly on the decisions in `docs/PRD.md`:

| File | Purpose |
|---|---|
| `src/urnings_tracker.py` | Production Urnings tracker (per-user, per-skill urn state) — prior injection, mastery criterion, escalation gate |
| `src/vocab_scheduler.py` | FSRS wrapper, scoped to vocab/memorization items only |
| `src/grammar_scheduler.py` | "Due" skills for grammar — Urnings-based (weak/uncertain first), not calendar-based |
| `src/streak.py` | Daily streak counter — deliberately independent of both trackers above |
| `src/daily_drill.py` | Assembles one day's quiz from due + new vocab and grammar items |

| `src/item_foundry.py` | Grammar `ContentProvider` (Novelty 1, PRD §9.70): EXAONE-3.5-7.8B-Instruct (4-bit) writes a per-skill misconception codebook and each item (sentence, key span, three distractors, each realising a different codebook misconception, optionally the learner's dominant one); verified by `kagas_annotator.py` and Qwen3-4B log-prob scoring, with critic revisions. No hand-written slots, lookups or fillers |
| `src/kagas_annotator.py` | KAGAS-style edit annotator: aligns kiwipiepy morphemes of the key and distractor sentences, classifies each edit by KAGAS category, and gives closed-class edits a stable code (`PART:에서>에`) |
| `src/misconception_tracker.py` | Per-learner misconception propensities from WHICH distractor is chosen: multinomial-logit choice among the offered distractors, updated online (PRD §9.70) |
| `src/vocab_provider.py` | Vocab `ContentProvider` — words/meanings from an admin-maintained curriculum, not generation; distractors are other curriculum meanings |
| `src/content_provider.py` | Combines the two into the single provider `daily_drill.py` actually needs once `vocab_ratio` mixes both types |
| `src/answer_processing.py` | The other half of the daily-drill lifecycle: processes a submitted answer — updates the Urnings skill urn and the misconception profile, checks escalation, calls `qwen_feedback`, updates the FSRS card, records streak completion |
| `src/question_bank.py` | Decouples generating items (slow, needs the LLM, batch) from serving them (fast, every drill assembly) — `QuestionBank.batch_generate()` + admin approve/reject/edit workflow, `BankContentProvider` serves only approved entries (sampled from the 85%-target kernel, preferring items that offer the learner's dominant misconception), no live LLM call at request time; `submit_answer` applies the Urnings selection correction |
| `src/qwen_feedback.py` | Fast + detailed (escalation-gated) wrong-answer feedback. Despite the file name it currently runs EXAONE-3.5-7.8B-Instruct (4-bit), which has no reasoning mode, so the detailed path is a longer prompt rather than reasoning. English is the active language; the Indonesian path has known quality issues (PRD §9.67) |
| `src/solver_check.py` | Verifier checks on generated items with Qwen3-4B: blind best-answer pick (letter, rotated over option orders, or cloze log-prob scoring) and log-likelihood "is the distractor also correct" check |
| `src/eval_harness.py` | Statistics for validation results: unique-item counts, Wilson 95% intervals, multi-seed aggregation, and a frozen dev/test split of curated items |
| `src/difficulty_heuristic.py` | Deterministic difficulty proxy (morpheme count + grammatical complexity) standing in for the KLUE-RoBERTa regression head, which is blocked by a missing calibration dataset |

Each file has runnable smoke tests in its `if __name__ == "__main__"` block
(`.venv/bin/python src/<file>.py`). `daily_drill.py` depends on real content
generation only through the `ContentProvider` protocol, mocked in its own
smoke test.

Current model per role (PRD §9.51–§9.70): item generation (`item_foundry.py`)
and feedback use `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` (4-bit);
`solver_check.py` (log-prob gate) uses `Qwen/Qwen3-4B` (bf16). The item
foundry's GPU validation is `python3 src/validate_item_foundry.py`. The
findings section below is the 2026-10-19 history of the template generator
(EXAONE-4.0-1.2B, Qwen3-1.7B/4B), which was removed on 2026-10-21; it is
kept for the record, and the files it names no longer exist.

### Content generation findings (2026-10-19)

KLUE-RoBERTa, EXAONE, and the KAGAS validator were, until now, only backed
by literature/vendor claims — never run against this project's actual
Korean grammar items. Tested for real on this machine's GPU (RTX 4070 Ti
SUPER, 16GB):

- **KLUE-RoBERTa-base**: loads and tokenizes Korean correctly. No
  fine-tuned regression head yet — no calibrated difficulty-labeled
  dataset exists to train one, and no amount of engineering fixes a
  missing dataset. Replaced for now, not left unimplemented:
  `src/difficulty_heuristic.py` is a deterministic proxy (morpheme count +
  grammatical complexity via `kiwipiepy`) with the same `(mu_hat,
  sigma_hat2)` interface, swappable for the trained encoder later with no
  downstream changes.
- **EXAONE-3.5-2.4B-Instruct**: its `trust_remote_code=True` custom
  modeling code is currently broken against every `transformers` release
  tested (4.56, 4.57, 5.0, 5.20 — different error each time). **Switched to
  EXAONE-4.0-1.2B**, natively supported (`exaone4` architecture, no remote
  code), which keeps the hybrid reasoning/non-reasoning mode the original
  choice was made for.
- **Generation quality, tested directly**: asked for a 에/에서 (particle)
  confusion item, EXAONE-4.0-1.2B produced a distractor built on `와` (a
  conjunction, unrelated grammar point) and an explanation that misnamed
  `에` as an accusative particle. `src/distractor_validator.py` catches
  exactly this failure automatically, via morphological tagging rather
  than trusting the generator's self-report.
- **Getting reliable generation out of this model took real debugging**,
  now built into `src/exaone_provider.py`: zero/one-shot JSON requests
  produced syntactically valid JSON with semantically wrong content
  (meta-commentary about the task, not an actual item); `float16`
  corrupted output with stray Cyrillic characters (fixed by switching to
  `bfloat16`); an open-ended prompt in non-reasoning mode failed near-
  instantly. The fix was a narrowly scaffolded prompt (explicit template +
  one worked example + a single slot to fill), which worked reliably
  (5/5 across five topics) — not a bigger model, not reasoning mode.
- **Qwen3 feedback** (`src/qwen_feedback.py`) went through three rounds,
  and the conclusion from round 2 was later reversed:
  1. Qwen3-1.7B asked to answer in Indonesian answered in Korean instead,
     ignoring the instruction entirely.
  2. A one-shot Indonesian example fixed language compliance (4/4) but
     introduced a worse failure: 3/4 trials leaked the example's own topic
     into the answer (explaining a restaurant-eating sentence as if it
     were about studying, because the example was about studying) —
     fluent Indonesian, wrong content, harder to catch than answering in
     the wrong language outright. This was recorded as "Indonesian
     eliminated, English-only final."
  3. That conclusion was challenged, correctly — the untested variable was
     model *size*, not whether the task was feasible at all.
     **Qwen3-4B**, same one-shot technique but with the example
     diversified to two different topics (reduces overfitting to one
     topic's words), got 4/4 correct and leak-free on both the fast path
     and the detailed/reasoning path. ~8GB VRAM, still comfortably inside
     the 16GB budget. `qwen_feedback.py` now supports `language="en"`/`"id"`
     on Qwen3-4B. The reasoning-mode (escalation) path still needs
     `max_new_tokens>=1200` — at 500 it runs out of budget *inside*
     `<think>` and produces zero usable output, in either language.
  One residual found while checking quality, not yet fixed: an
  illustrative example Qwen wrote for itself in the English detailed path
  contained a conjugation error (들어요 → 듣어요) — a different pipeline
  from the EXAONE conjugation fix below, not covered by it.
- **Grammar generation was redesigned to remove conjugation/particle risk
  entirely**, not just validate around it: `kiwipiepy`'s analyzer is too
  permissive to catch bad conjugations (it happily re-parses "마시요" as if
  valid — built for noisy text, not as a grammar checker). Fix: the LLM is
  now asked only for a dictionary-form verb lemma (genuine vocabulary
  variety); the particle's correctness and the verb's conjugation are both
  built deterministically (`kiwi.join()`, correct even for irregulars:
  마시다→마셔요, 보다→봐요, 하다→해요). Result: 4/5 and 5/5 generations
  succeeded with zero conjugation or particle errors — the LLM can no
  longer get either wrong, because it's no longer asked to produce either.

`make_figures.py` fails loudly if any text overflows its box — layout bugs are
errors, not something to eyeball.

## Files

| File | Purpose |
|---|---|
| `src/simulate_trackers.py` | Elo / Parallel Elo / Urnings / Glicko comparison under adaptive selection + dynamic item calibration (static ability) |
| `src/simulate_drift.py` | Same trackers under a *drifting* (improving) ability — tests whether each tracker keeps up with real learning gain, not just precision on a fixed target |
| `src/figlib.py` | Drawing helpers with automatic text-fit verification |
| `src/make_figures.py` | The six paper figures |
| `src/content.py` | All paper text and the reference list |
| `src/build_docx.py` | IEEE DOCX builder (two columns, full-width figures, real sub/superscripts) |

## The simulation result

300 learners, 400 items, 120,000 responses, Rasch generating model, adaptive
selection with simultaneous item calibration — the regime a generative item pool
forces. True SD of ability = 1.04.

| Tracker | Terminal SD | Correlation | RMSE |
|---|---|---|---|
| Elo | 2.69 (still rising) | 0.97 | 1.70 |
| Parallel Elo | 1.71 | 0.98 | 0.73 |
| **Urnings** | **1.11** | 0.92 | **0.43** |

Elo keeps ranking learners correctly while the scale it reports them on drifts —
which is why the failure is easy to miss in production. Urn-size sensitivity
(16/32/64/128 balls) gives RMSE 0.59 / 0.45 / 0.35 / 0.30.

## Citations

All references were verified against the Crossref API — authors, venue,
volume, pages, year, DOI. No arXiv or MDPI sources in the literature review;
the two model technical reports (EXAONE, Qwen3) and the KLUE benchmark are cited
as model/resource documentation, not as review literature.

## Status

The paper is a design-and-protocol contribution: architecture, rationale, three
novelties, and a pre-registered evaluation plan. The tracker simulation is real
and reproducible. The human study described in Section V has **not** been run —
no learning-outcome results are reported or implied.
