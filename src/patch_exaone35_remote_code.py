"""
Fixes a real bug in LGAI-EXAONE's own `trust_remote_code` custom modeling
file for the EXAONE-3.5 family (2.4B and 7.8B confirmed, PRD §9.50), found
while investigating why neither model could run in this environment at
all (3 different transformers versions tried first -- 4.46.3, 4.57.1,
5.20.1 -- all failed with DIFFERENT errors, which looked like "no
compatible version exists" until the actual bug was isolated).

Root cause, found by reading the actual installed function signature
(not guessed): the repo's modeling_exaone.py calls
    create_causal_mask(config=..., input_embeds=..., cache_position=..., ...)
but transformers 5.20.1's real create_causal_mask() signature is
    create_causal_mask(config, inputs_embeds, attention_mask, past_key_values,
                        position_ids=None, ...)  # no cache_position at all
Two separate problems in ONE call: (1) "input_embeds" vs "inputs_embeds"
-- almost certainly a typo/rename the repo never updated for, (2)
"cache_position" was removed from the function entirely (position_ids,
already derived from cache_position earlier in the same function, covers
it). Fixing just the name alone is not enough; both edits are required.

This patches the LOCAL cache copy at
~/.cache/huggingface/modules/transformers_modules/.../modeling_exaone.py
-- NOT the installed transformers library, and NOT the Hub repo itself.
That cache is keyed by model+revision and is NOT guaranteed to survive a
fresh environment, a cache clear, or `trust_remote_code` re-fetching the
file (observed to happen opportunistically even when the remote content
is unchanged) -- run this AFTER first loading the tokenizer/model once
(so the file exists locally) and again any time generation starts
failing with the same create_causal_mask TypeError.

Usage: .venv/bin/python src/patch_exaone35_remote_code.py
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

_CACHE_GLOB = str(Path.home() / ".cache/huggingface/modules/transformers_modules"
                   "/LGAI_hyphen_EXAONE/EXAONE_hyphen_3_dot_5*/*/modeling_exaone.py")


def patch_file(path: str) -> bool:
    text = Path(path).read_text()
    original = text
    text = text.replace("input_embeds=inputs_embeds,", "inputs_embeds=inputs_embeds,")
    text = re.sub(r"[ \t]*cache_position=cache_position,\n", "", text)
    if text != original:
        Path(path).write_text(text)
        return True
    return False


if __name__ == "__main__":
    matches = glob.glob(_CACHE_GLOB)
    if not matches:
        print("No cached EXAONE-3.5 modeling_exaone.py found yet -- load the "
              "tokenizer/model at least once first (trust_remote_code=True) "
              "so transformers fetches it, then re-run this script.")
    for path in matches:
        changed = patch_file(path)
        print(f"{'patched' if changed else 'already OK / nothing to change'}: {path}")
