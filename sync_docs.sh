#!/bin/bash

echo "Syncing notebooks to docs..."

if [ -d "src/notebook" ]; then
    echo "Copying notebooks from src/notebook to docs/implementation..."
    cp -v src/notebook/*.ipynb docs/implementation/ 2>/dev/null || echo "  No .ipynb files found in src/notebook"
fi

echo "Sync complete."
