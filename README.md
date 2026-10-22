# KO-DRILL

Personalised Korean grammar/vocabulary drill quizzes: LLM-generated multiple-choice
items, with difficulty adapted per learner per skill via Urnings.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Structure

- `src/kodrill/generation/` — item generation (EXAONE writes each item: sentence,
  distractors, per-option reasoning), driven by `Curriculum.items`
- `src/kodrill/urnings/` — Urnings mastery tracking and adaptive scheduling
- `src/kodrill/drill/` — orchestration: assembling a drill session, serving items,
  processing answers
- `src/extras/` — vocab spaced-repetition, streaks, live answer feedback
- `data/` — reference material only (curriculum JSON/docx), not code; loaded via
  `Curriculum.from_json` and handed to the LLM as context for what to generate

## Run

Each module has a smoke test in its `if __name__ == "__main__"` block, run from
the repo root:

```bash
PYTHONPATH=src .venv/bin/python -m kodrill.generation.item_foundry
```

Swap in the dotted module path for any other file under `src/`.
