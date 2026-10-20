"""
Follow-up to investigate_kolla_corpus.py (PRD §9.58/§9.59): that script's
own taxonomy survey found R:SPELL as KoLLA's single largest error-type tag
by far (1986 of 3121 annotations, PRD §9.59's real run) -- far more data
than any particle-substitution category (the largest, 에->에서, was only
55). User asked directly whether KAGAS's edit-operation categories
(INS/DEL/WS/WO/SPELL/PUNCT/SHORT, excluded from this project's 11 skills
because, per content.py, only the 7 POS-class categories were judged
"relevant") could actually be trained/corpus-grounded rather than assumed
infeasible for this item format.

This script answers the SPELL-specific sub-question concretely: what do
these 1986 corrections actually look like (systematic confusions a cloze
MCQ item could test, e.g. 되/돼, vs. one-off typos unique to a single
essay), and is there a learnable PATTERN with enough repetition per
pattern to either (a) curate a new skill's distractor pairs from real
frequency data, or (b) fine-tune a small classifier given 1986 is a much
larger dataset than the particle-category counts PRD §9.59 judged too
sparse for that.

Reuses the already-downloaded corpus (kolla_corpus_download/,
investigate_kolla_corpus.py's download_and_extract()) rather than
re-fetching -- run that script first if this directory doesn't exist yet.

Could NOT be run in the cloud sandbox this was written in (zenodo.org
blocked, same as before) -- run on a machine that already has
kolla_corpus_download/ populated.

Usage: python3 src/investigate_kolla_spell.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from investigate_kolla_corpus import download_and_extract, find_m2_files, parse_m2


def main() -> None:
    if not find_m2_files():
        print("kolla_corpus_download/ not found -- downloading (same as "
              "investigate_kolla_corpus.py does)...")
        download_and_extract()
    m2_files = find_m2_files()
    if not m2_files:
        print("No .m2 file found -- see investigate_kolla_corpus.py's own error message.")
        return

    all_annotations = []
    for path in m2_files:
        all_annotations.extend(parse_m2(path))

    spell_pairs = [(orig, corr) for etype, orig, corr in all_annotations if etype == "R:SPELL"]
    print(f"=== Total R:SPELL annotations: {len(spell_pairs)} ===\n")

    pair_counts = Counter(spell_pairs)
    n_unique_pairs = len(pair_counts)
    n_repeated = sum(1 for count in pair_counts.values() if count >= 2)
    print(f"Unique (original, correction) pairs: {n_unique_pairs}")
    print(f"Pairs that repeat (count >= 2) -- i.e. a SYSTEMATIC confusion, "
          f"not a one-off typo: {n_repeated}\n")

    print("=== Top 40 most frequent SPELL corrections ===")
    for (orig, corr), count in pair_counts.most_common(40):
        print(f"  {orig!r} -> {corr!r}: {count}")

    print("\n=== Sample of SINGLETON corrections (count==1), first 20 -- to judge "
          "whether the long tail is systematic confusions or essay-specific typos ===")
    singletons = [(pair, count) for pair, count in pair_counts.items() if count == 1][:20]
    for (orig, corr), count in singletons:
        print(f"  {orig!r} -> {corr!r}")

    print("\nDecision rule for this investigation: a handful of pairs with count>=10 "
          "would be strong, curatable, frequency-grounded distractor candidates for a "
          "NEW 'spelling confusion' skill (same format as this project's other 11 -- "
          "cloze blank, correct vs. commonly-confused form as the distractor) -- that is "
          "option A from before, just for a new category instead of re-curating an "
          "existing one. If the TOTAL corpus (1986) is large but spread across hundreds "
          "of near-unique pairs with low per-pair repetition, that argues the errors are "
          "mostly idiosyncratic typos rather than a small set of systematic confusions "
          "worth building a skill template around -- check both numbers above, not just "
          "the total.")


if __name__ == "__main__":
    main()
