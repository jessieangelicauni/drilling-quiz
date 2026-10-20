#!/usr/bin/env bash
# Safe commit helper -- shows what's changed before staging anything, never
# stages blindly (no `git add -A`), and always reviews the diff of what it's
# about to commit before committing it.
#
# Usage:
#   tools/commit.sh                        # just show status + diff, stage nothing
#   tools/commit.sh "commit message" file1 [file2 ...]
#
# With no arguments: prints `git status` and a summary diff so you can see
# exactly what changed, then exits -- nothing is staged or committed. Re-run
# with a message and the specific files you want once you've reviewed it.
#
# With a message + files: stages exactly those files (never anything else),
# shows the staged diff for one more look, asks for y/n confirmation, then
# commits and shows the result. Never runs `git add -A`/`git add .` and
# never force-pushes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ $# -eq 0 ]]; then
    echo "=== git status ==="
    git status
    echo
    echo "=== unstaged diff (stat only) ==="
    git diff --stat
    echo
    echo "=== untracked files ==="
    git status --porcelain | grep '^??' | sed 's/^?? /  /' || echo "  (none)"
    echo
    echo "Nothing staged or committed. Re-run as:"
    echo "  tools/commit.sh \"your commit message\" path/to/file1 path/to/file2 ..."
    exit 0
fi

MESSAGE="$1"
shift
FILES=("$@")

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: no files given. Pass the specific files to commit, e.g.:" >&2
    echo "  tools/commit.sh \"fix typo\" src/foo.py docs/PRD.md" >&2
    exit 1
fi

echo "=== staging ==="
for f in "${FILES[@]}"; do
    if [[ ! -e "$f" ]]; then
        echo "Error: '$f' does not exist -- typo in the path?" >&2
        exit 1
    fi
    echo "  + $f"
done
git add -- "${FILES[@]}"

echo
echo "=== staged diff ==="
git diff --cached

echo
read -r -p "Commit these ${#FILES[@]} file(s) with message: \"$MESSAGE\"? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted -- changes are staged but not committed. Run 'git reset' to unstage."
    exit 1
fi

git commit -m "$MESSAGE"

echo
echo "=== result ==="
git status
git log --oneline -1
