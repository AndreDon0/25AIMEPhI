#!/usr/bin/env bash
# Sync this fork with upstream (ShadarRim/25AIMEPhI).
# Uses merge, not pull. Students/ is kept out via .gitattributes merge driver.

set -e

UPSTREAM_URL="https://github.com/ShadarRim/25AIMEPhI.git"
UPSTREAM_REMOTE="upstream"
BRANCH="${1:-main}"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: not inside a git repository." >&2
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
git merge "$UPSTREAM_REMOTE/$BRANCH" --no-edit

echo "Done. Push with: git push origin $BRANCH"
