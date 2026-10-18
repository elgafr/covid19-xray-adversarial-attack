#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Synchronizing research artifacts into docs/site_docs/..."

mkdir -p "$SCRIPT_DIR/site_docs/implementation"
mkdir -p "$SCRIPT_DIR/site_docs/images"
mkdir -p "$SCRIPT_DIR/site_docs/publications"

if [ -d "$REPO_ROOT/notebooks" ]; then
    cp -f "$REPO_ROOT/notebooks/"*.ipynb "$SCRIPT_DIR/site_docs/implementation/" 2>/dev/null || true
fi

if [ -d "$REPO_ROOT/results/figures" ]; then
    cp -f "$REPO_ROOT/results/figures/"*.png "$SCRIPT_DIR/site_docs/images/" 2>/dev/null || true
fi

if [ -d "$REPO_ROOT/journal" ]; then
    cp -f "$REPO_ROOT/journal/"*.pdf "$SCRIPT_DIR/site_docs/publications/" 2>/dev/null || true
fi

echo "Synchronization complete."
