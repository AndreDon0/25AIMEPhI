#!/usr/bin/env bash
# Sync this fork with upstream (ShadarRim/25AIMEPhI).
#
# After merging upstream, the Students/ folder is force-reset to its pre-merge
# state in this fork (including remaining absent if we deleted it). Any upstream
# additions/modifications/deletions under Students/ are discarded. This is done
# in the script itself because .gitattributes merge drivers only run on real
# 3-way conflicts and therefore cannot block non-conflicting upstream changes.
#
# Usage: ./sync-fork.sh [branch]   (defaults to main)

set -euo pipefail

UPSTREAM_URL="https://github.com/ShadarRim/25AIMEPhI.git"
UPSTREAM_REMOTE="upstream"
BRANCH="${1:-main}"
EXCLUDE_PATH="Students"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash changes first." >&2
  exit 1
fi

if ! git remote get-url "$UPSTREAM_REMOTE" &>/dev/null; then
  echo "Adding remote $UPSTREAM_REMOTE..."
  git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
else
  echo "Remote $UPSTREAM_REMOTE already exists."
fi

echo "Fetching from $UPSTREAM_REMOTE..."
git fetch "$UPSTREAM_REMOTE"

echo "Merging $UPSTREAM_REMOTE/$BRANCH into current branch..."
set +e
git merge --no-ff --no-commit "$UPSTREAM_REMOTE/$BRANCH"
set -e

if [[ ! -f .git/MERGE_HEAD ]]; then
  echo "Already up to date."
  exit 0
fi

unmerged_outside="$(git diff --name-only --diff-filter=U \
  -- . ":(exclude)$EXCLUDE_PATH" ":(exclude)$EXCLUDE_PATH/**" || true)"
if [[ -n "$unmerged_outside" ]]; then
  echo "Error: merge conflicts outside $EXCLUDE_PATH/. Resolve them manually:" >&2
  echo "$unmerged_outside" >&2
  echo "Run 'git merge --abort' to cancel the in-progress merge." >&2
  exit 1
fi

echo "Resetting $EXCLUDE_PATH/ to pre-merge state..."
git rm -rf --cached --ignore-unmatch -- "$EXCLUDE_PATH" >/dev/null 2>&1 || true
rm -rf -- "$EXCLUDE_PATH"
git checkout HEAD -- "$EXCLUDE_PATH" 2>/dev/null || true

echo "Committing merge..."
git commit --no-edit

echo "Done. Push with: git push origin $BRANCH"
