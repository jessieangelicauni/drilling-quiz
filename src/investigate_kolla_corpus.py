"""
Investigation script (not production code) for mining KoLLA v2.0 (Song et
al. 2026, "Enriching the Korean learner corpus for grammatical error
correction and writing assessment") for real, attested particle-error
frequency data -- to answer two questions before committing further effort:

1. Is there enough real data, per particle-substitution type, to fine-tune
   a small classifier (option B, discussed with the user) -- or only
   enough to inform a frequency-based re-curation of exaone_provider.py's
   particle_loc/particle_obj distractors (option A, lower effort)?
2. What does the corpus's actual error-type taxonomy look like? (Not
   verified from the paper alone -- the paper describes the ANNOTATION
   SCHEME in prose, "omission/substitution/addition/ordering", but the
   exact M2 error_type tag strings used in the real file were not
   confirmed by reading the PDF; this script surveys them directly rather
   than assuming a specific tag format.)

Scope, set by the paper itself (read directly, not assumed): KoLLA is
PARTICLE-ERROR-ONLY. It cannot surface new skill categories for
honorific/tense_past/irregular_conj/adj_relativizer/verb_choice/
noun_counter/mod_adverb -- only particle_loc/particle_obj.

Could NOT be run in the cloud sandbox this was written in: zenodo.org is
blocked by that session's network policy (confirmed via the agent-proxy
status endpoint -- a policy denial, not transient, same as
huggingface.co). Run this on a machine with real internet access.

License: KoLLA v2.0 is released under GPLv3 (per the paper's Data
Availability statement) -- compatible with inspecting/mining it for
frequency statistics to inform this project's own (differently licensed)
content; re-distributing the corpus itself is a separate question not
addressed here.

Usage: python3 src/investigate_kolla_corpus.py
(needs: pip install requests, if not already available)
"""
from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import requests

_ZENODO_RECORD_API = "https://zenodo.org/api/records/15287128"
_DOWNLOAD_DIR = Path("kolla_corpus_download")

# The particle set this project's particle_loc/particle_obj skills actually
# use (exaone_provider.py SKILL_SPECS) -- only substitutions involving these
# are directly actionable for this project; others are survey-only context.
_PROJECT_PARTICLES = {"이", "가", "을", "를", "은", "는", "에", "에서", "과", "와", "로", "으로"}

_M2_ANNOTATION_RE = re.compile(
    r"^A (?P<start>-?\d+) (?P<end>-?\d+)\|\|\|(?P<etype>[^|]*)\|\|\|(?P<correction>[^|]*)\|\|\|")


def download_and_extract() -> list[Path]:
    _DOWNLOAD_DIR.mkdir(exist_ok=True)
    print(f"Fetching Zenodo record metadata: {_ZENODO_RECORD_API}")
    meta = requests.get(_ZENODO_RECORD_API, timeout=30).json()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError("Zenodo record metadata returned no files -- API shape may have "
                            "changed, or the record is restricted; check the record page "
                            "manually at https://zenodo.org/records/15287128")
    print(f"Found {len(files)} file(s) in the record:")
    local_paths = []
    for f in files:
        url = f["links"]["self"]
        name = f["key"]
        print(f"  downloading {name} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        local_path = _DOWNLOAD_DIR / name
        local_path.write_bytes(resp.content)
        local_paths.append(local_path)
        if name.endswith(".zip"):
            with zipfile.ZipFile(local_path) as zf:
                zf.extractall(_DOWNLOAD_DIR)
                print(f"    extracted {len(zf.namelist())} entries")
    return local_paths


def find_m2_files() -> list[Path]:
    return list(_DOWNLOAD_DIR.rglob("*.m2")) + list(_DOWNLOAD_DIR.rglob("*.M2"))


def parse_m2(path: Path) -> list[tuple[str, str, str]]:
    """Returns (error_type, original_span, correction_text) tuples.

    FIX (v2, after the first run showed only 2/4464 matches -- the bug was
    here, not a lack of data): the original version matched `correction`
    directly against a short particle set, but M2's correction field holds
    the WHOLE corrected span/word (e.g. "학교에서"), not just the particle
    ("에서") -- almost every real particle-substitution instance was
    invisible to that filter. Fixed by tracking each `S` line's tokens
    (M2 annotations reference them by token index) so the ORIGINAL span can
    actually be recovered and diffed against the correction, not just the
    correction read in isolation.
    """
    results = []
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("S "):
            tokens = line[2:].split()
            continue
        m = _M2_ANNOTATION_RE.match(line)
        if not m:
            continue
        start, end = int(m.group("start")), int(m.group("end"))
        correction = m.group("correction").strip()
        if start < 0 or end < 0 or end > len(tokens) or end <= start:
            continue  # -NONE-/insertion-only annotations, or out-of-range -- not a substitution
        original_span = "".join(tokens[start:end])
        results.append((m.group("etype").strip(), original_span, correction))
    return results


# Sorted longest-first so e.g. "에서" is tried before "에" -- a naive
# character-diff (first version of this function) breaks exactly on this
# case: "에" is a literal prefix of "에서", so blind longest-common-prefix
# diffing of "학교에"/"학교에서" consumes "에" as part of the "unchanged"
# prefix and leaves an empty "before" particle, silently dropping the most
# basic 에/에서 confusion this project's own particle_loc skill is built
# around. Fixed by matching against a known-suffix list instead of diffing.
_KNOWN_PARTICLE_SUFFIXES = sorted([
    "에게서", "한테서", "으로서", "으로써", "에다가",
    "에서", "에게", "한테", "로서", "로써", "으로", "에다",
    "이랑", "하고", "이나", "께서",
    "이", "가", "을", "를", "은", "는", "의", "도", "만", "에", "께", "로", "와", "과", "나",
], key=len, reverse=True)


def _strip_particle(word: str) -> tuple[str, str | None]:
    for particle in _KNOWN_PARTICLE_SUFFIXES:
        if word.endswith(particle) and len(word) > len(particle):
            return word[: -len(particle)], particle
    return word, None


def extract_particle_change(original_span: str, correction: str) -> tuple[str, str] | None:
    """Isolates a particle substitution, e.g. ("학교에", "학교에서") ->
    ("에", "에서"), by matching known particle suffixes rather than diffing
    characters -- only returns a pair when the NOUN STEM is identical on
    both sides (so a genuine word change, not just a particle change,
    is correctly excluded rather than misread as one).
    """
    orig_stem, orig_particle = _strip_particle(original_span)
    corr_stem, corr_particle = _strip_particle(correction)
    if orig_particle is None or corr_particle is None:
        return None
    if orig_stem != corr_stem or orig_particle == corr_particle:
        return None
    return orig_particle, corr_particle


def main() -> None:
    if not find_m2_files():
        download_and_extract()
    m2_files = find_m2_files()
    if not m2_files:
        print("No .m2 file found after download -- inspect kolla_corpus_download/ manually; "
              "the corpus may use a different annotation file format than M2 "
              "(the paper's Appendix A says M2, but this has not been directly confirmed).")
        return

    print(f"\nFound {len(m2_files)} M2 file(s): {[str(p) for p in m2_files]}\n")

    all_annotations = []
    for path in m2_files:
        all_annotations.extend(parse_m2(path))

    print(f"=== Total annotations parsed: {len(all_annotations)} ===\n")

    print("=== Error type tag frequency (unfiltered survey -- actual taxonomy, not assumed) ===")
    etype_counts = Counter(etype for etype, _, _ in all_annotations)
    for etype, count in etype_counts.most_common(30):
        print(f"  {etype}: {count}")

    print("\n=== Particle-change pairs (before -> after), via original/correction diff ===")
    all_changes = Counter()
    project_changes = Counter()
    unresolved = 0
    for etype, original_span, correction in all_annotations:
        if "ADP" not in etype:  # ADP = adposition/particle, per the taxonomy survey above
            continue
        change = extract_particle_change(original_span, correction)
        if change is None:
            unresolved += 1
            continue
        all_changes[change] += 1
        if change[0] in _PROJECT_PARTICLES or change[1] in _PROJECT_PARTICLES:
            project_changes[change] += 1

    print(f"(ADP-tagged annotations where the diff could NOT isolate a clean short suffix: "
          f"{unresolved} -- likely whole-word changes alongside the particle, not pure "
          f"substitutions; excluded rather than guessed)\n")

    print("All particle-change pairs found (any length <=3 chars, not just this project's set):")
    for (before, after), count in all_changes.most_common(30):
        print(f"  {before!r} -> {after!r}: {count}")

    print("\nProject-relevant subset (involves a particle particle_loc/particle_obj already use):")
    if project_changes:
        for (before, after), count in project_changes.most_common():
            print(f"  {before!r} -> {after!r}: {count}")
    else:
        print("  none -- the particles this project targets may not be the ones KoLLA's "
              "learners actually struggled with; see the full list above instead.")

    total_particle_instances = sum(project_changes.values())
    print(f"\nTotal project-relevant particle-change instances: {total_particle_instances}")
    print("\nDecision rule for this investigation: fine-tuning (option B) needs enough "
          "instances PER particle type to avoid overfitting a classifier (rule of thumb: "
          "at least several dozen per class, ideally 100+) -- if most particles above have "
          "fewer than ~30 instances, option A (frequency-informed re-curation, no training) "
          "is the realistic path; option B would need a larger or supplementary corpus.")


if __name__ == "__main__":
    main()
