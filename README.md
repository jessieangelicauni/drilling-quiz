# KO-DRILL — IEEE paper, figures, and simulation

**Mulai dari sini:** [`docs/PRD.md`](docs/PRD.md) — semua alasan desain (kenapa Parallel Elo bukan DKT, kenapa EXAONE/Qwen3/KLUE-RoBERTa, tiga novelty riset, apa yang sengaja TIDAK dikerjakan) jadi satu acuan.

**Literature review:** [`docs/literature-review-15.md`](docs/literature-review-15.md) — 15 curated sources (2024–2026, IEEE/Springer/Elsevier/Wiley/Taylor & Francis/SAGE only, no arXiv/MDPI) with DOI links and explicit research-gap mapping.

Personalised Korean drill quizzes: a psychometrically grounded LLM pipeline.

## Output

- `paper/KO-DRILL_IEEE.docx` — the paper, IEEE two-column format, 11 pages, 36 references
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

| `src/distractor_validator.py` | KAGAS-aligned validator (`kiwipiepy` morphological analysis) — confirms a distractor option actually parses as the particle/tag it's meant to test |
| `src/exaone_provider.py` | Real `ContentProvider` for grammar items — EXAONE-4.0-1.2B generation, gated by `distractor_validator.py`, wired and tested through `daily_drill.py` end-to-end |
| `src/vocab_provider.py` | Vocab `ContentProvider` — words/meanings from an admin-maintained curriculum, not generation; distractors are other curriculum meanings |
| `src/content_provider.py` | Combines the two into the single provider `daily_drill.py` actually needs once `vocab_ratio` mixes both types |
| `src/qwen_feedback.py` | Fast + detailed (escalation-gated) wrong-answer feedback via Qwen3-1.7B — English-only by tested decision, not a TODO (see below) |
| `src/difficulty_heuristic.py` | Deterministic difficulty proxy (morpheme count + grammatical complexity) standing in for the KLUE-RoBERTa regression head, which is blocked by a missing calibration dataset |

Each file has runnable smoke tests in its `if __name__ == "__main__"` block
(`.venv/bin/python src/<file>.py`). `daily_drill.py` depends on real content
generation only through the `ContentProvider` protocol, mocked in its own
smoke test — the generator itself (EXAONE-4.0-1.2B) was smoke-tested
standalone (see "Content generation findings" below), not yet wired in as
a `ContentProvider` implementation.

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
- **Qwen3-1.7B feedback** (`src/qwen_feedback.py`): asked to explain a
  mistake in Indonesian, it answered in Korean instead, ignoring the
  language instruction entirely — fixed with a stricter "Answer ONLY in
  English" prompt, so this feature is English-only for now, not the
  bilingual ID/EN originally planned. Its reasoning-mode (escalation) path
  at `max_new_tokens=500` ran out of budget *inside* `<think>` and
  produced zero usable output; `max_new_tokens=1200` reliably finishes
  (~1000 tokens, ~12s) with a genuinely more detailed explanation than the
  fast path — so the escalation path's extra cost is real, but only pays
  off if the token budget is set high enough.
- **Indonesian feedback was tried again and eliminated, not deferred**: a
  one-shot Indonesian worked example fixed language compliance (4/4), but
  3/4 trials then leaked the example's own topic into the answer (e.g.
  explaining a restaurant-eating sentence as if it were about studying,
  because the example was about studying) — fluent Indonesian, wrong
  content, harder to catch than answering in the wrong language outright.
  `qwen_feedback.py` is English-only by this tested result.
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

All 36 references were verified against the Crossref API — authors, venue,
volume, pages, year, DOI. No arXiv or MDPI sources in the literature review;
the two model technical reports (EXAONE, Qwen3) and the KLUE benchmark are cited
as model/resource documentation, not as review literature.

## Status

The paper is a design-and-protocol contribution: architecture, rationale, three
novelties, and a pre-registered evaluation plan. The tracker simulation is real
and reproducible. The human study described in Section V has **not** been run —
no learning-outcome results are reported or implied.
