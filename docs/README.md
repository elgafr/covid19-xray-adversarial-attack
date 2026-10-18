# Documentation Website (MkDocs Material)

This directory contains the self-contained configuration, dependencies, synchronization script, and Markdown content for the project's documentation website deployed at:

[https://hisyam99.github.io/covid19-xray-adversarial-attack/](https://hisyam99.github.io/covid19-xray-adversarial-attack/)

## Directory Structure

- `mkdocs.yml` — Main MkDocs Material configuration (`docs_dir: site_docs`).
- `requirements.txt` — Python dependencies required to build the documentation website.
- `sync_docs.sh` — Script to synchronize the latest notebook (`../notebooks/`), figures (`../results/figures/`), and journal PDF (`../journal/`) into `site_docs/`.
- `site_docs/` — Documentation Markdown source pages, custom overrides, stylesheets, MathJax scripts, and static assets.

## Local Preview & Build Commands

From the repository root:

```bash
pip install -r docs/requirements.txt
./docs/sync_docs.sh
mkdocs serve -f docs/mkdocs.yml --dev-addr 127.0.0.1:8000
```

To run a strict production build:

```bash
mkdocs build -f docs/mkdocs.yml --strict -d ../site
```
