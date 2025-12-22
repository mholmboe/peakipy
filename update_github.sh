#!/usr/bin/env bash
# Simple helper to add/commit and force-push the current main branch to origin.
# Make sure you’re certain you want to replace the remote main.
# Usage: ./update_github.sh "Commit message"

set -euo pipefail

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
  echo "Currently on branch '$BRANCH'. Switch to 'main' before pushing." >&2
  exit 1
fi

MSG="${1:-Update}"

# Commit changes if any
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "Staging and committing changes with message: \"$MSG\""
  git add -A
  git commit -m "$MSG"
else
  echo "No changes to commit."
fi

echo "Force-pushing '$BRANCH' to origin..."
git push -f origin "$BRANCH"
echo "Done."
