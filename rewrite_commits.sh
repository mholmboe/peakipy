#!/usr/bin/env bash
# Rewrite ALL commit messages in the repository history to a single message.
# WARNING: This rewrites Git history and changes all commit hashes!
# Only use on personal repos where no one else has cloned/pulled.
#
# Usage: ./rewrite_commits.sh "New commit message"

set -euo pipefail

usage() {
  cat <<EOF
Usage:
  ./rewrite_commits.sh "New commit message"

This script will:
  1. Rewrite ALL commit messages in history to the specified message
  2. Force-push to origin (if on main branch)

WARNING: This is a destructive operation that rewrites Git history!
         All commit hashes will change. Only use on personal repositories.

Options:
  -h, --help    Show this help
  --dry-run     Show what would happen without making changes
EOF
}

DRY_RUN=false
MSG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [ -z "$MSG" ]; then
        MSG="$1"
      else
        echo "Error: unexpected argument: $1" >&2
        usage
        exit 2
      fi
      shift
      ;;
  esac
done

if [ -z "$MSG" ]; then
  echo "Error: commit message is required" >&2
  usage
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "========================================"
echo "  COMMIT MESSAGE REWRITER"
echo "========================================"
echo ""
echo "New message for ALL commits: \"$MSG\""
echo "Current branch: $BRANCH"
echo "Total commits: $(git rev-list --count HEAD)"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would rewrite all commits with message: \"$MSG\""
  echo "[DRY RUN] Would force-push to origin/$BRANCH"
  exit 0
fi

# Confirm before proceeding
read -p "⚠️  This will rewrite ALL commit history! Continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo "Rewriting commit messages..."

# Use git filter-branch to rewrite all commit messages
# The --force flag allows re-running if refs/original exists
git filter-branch --force --msg-filter "echo '$MSG'" -- --all

echo ""
echo "Cleaning up refs/original backup..."
rm -rf .git/refs/original/

if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo ""
  read -p "Force-push to origin/$BRANCH? (y/N): " push_confirm
  if [[ "$push_confirm" =~ ^[Yy]$ ]]; then
    echo "Force-pushing to origin/$BRANCH..."
    git push -f origin "$BRANCH"
    echo "Done!"
  else
    echo "Skipped push. Run 'git push -f origin $BRANCH' when ready."
  fi
else
  echo ""
  echo "Not on main/master branch. Run 'git push -f origin $BRANCH' to push changes."
fi

echo ""
echo "✅ All commits now have message: \"$MSG\""
