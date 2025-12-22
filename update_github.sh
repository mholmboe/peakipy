#!/usr/bin/env bash
# Simple helper to force-push the current main branch to origin.
# Make sure you’ve committed your local changes first, and only use it when you’re certain you want to replace the remote main.
# Usage: ./update_github.sh

set -euo pipefail

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
  echo "Currently on branch '$BRANCH'. Switch to 'main' before pushing." >&2
  exit 1
fi

echo "Force-pushing '$BRANCH' to origin..."
git push -f origin "$BRANCH"
echo "Done."
