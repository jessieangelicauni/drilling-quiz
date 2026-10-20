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


def parse_m2(path: Path) -> list[tuple[str, str]]:
    """Returns a list of (error_type, correction_text) tuples found in one
    M2 file -- does NOT assume the exact tag vocabulary in advance.
    """
    results = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _M2_ANNOTATION_RE.match(line)
        if m:
            results.append((m.group("etype").strip(), m.group("correction").strip()))
    return results


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
    etype_counts = Counter(etype for etype, _ in all_annotations)
    for etype, count in etype_counts.most_common(30):
        print(f"  {etype}: {count}")

    print("\n=== Particle-substitution-specific counts (project-relevant subset) ===")
    # Correction text for a particle-substitution error is typically just
    # the correct particle itself (single short token) -- filtered
    # defensively rather than assuming the exact etype string.
    particle_corrections = Counter(
        correction for etype, correction in all_annotations
        if correction in _PROJECT_PARTICLES
    )
    if particle_corrections:
        for particle, count in particle_corrections.most_common():
            print(f"  -> {particle}: {count} instances")
    else:
        print("  none matched by this heuristic -- inspect the etype survey above and "
              "adjust the filter; the correction-text field may not isolate single "
              "particles the way this script assumed.")

    total_particle_instances = sum(particle_corrections.values())
    print(f"\nTotal project-relevant particle-correction instances: {total_particle_instances}")
    print("\nDecision rule for this investigation: fine-tuning (option B) needs enough "
          "instances PER particle type to avoid overfitting a classifier (rule of thumb: "
          "at least several dozen per class, ideally 100+) -- if most particles above have "
          "fewer than ~30 instances, option A (frequency-informed re-curation, no training) "
          "is the realistic path; option B would need a larger or supplementary corpus.")


if __name__ == "__main__":
    main()
