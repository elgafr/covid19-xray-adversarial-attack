# Documentation Build & GitHub Pages Deployment

This documentation site is encapsulated inside the `docs/` module and built with **Material for MkDocs** using the latest stable versions of its core packages and plugins:

| Component / Plugin | Python Package | Version | Primary Role |
|:-------------------|:---------------|:-------:|:-------------|
| **Core Site Generator** | `mkdocs` | `>= 1.6.1` | Static documentation site generator |
| **UI Theme** | `mkdocs-material` | `>= 9.7.7` | Material Design theme with light/dark palette, content tabs, grid cards, and instant navigation |
| **Jupyter Integration** | `mkdocs-jupyter` | `>= 0.26.3` | Direct rendering of `.ipynb` notebooks into documentation pages |
| **Image Lightbox** | `mkdocs-glightbox` | `>= 0.5.2` | Interactive zoom, pan, and slide gallery for all figures |
| **Asset Minifier** | `mkdocs-minify-plugin` | `>= 0.8.0` | HTML, CSS, and JavaScript minification for production builds |
| **Git Revision Date** | `mkdocs-git-revision-date-localized-plugin` | `>= 1.6.0` | Localized creation and modification timestamps in CI/CD |
| **Markdown Extensions** | `pymdown-extensions` | `>= 12.3` | MathJax (`arithmatex`), `superfences` (Mermaid), `tabbed`, and `admonition` support |

---

## 1. Synchronize Research Artifacts (`docs/sync_docs.sh`)

Before serving or building the documentation, run `./docs/sync_docs.sh` to synchronize the latest notebook (`notebooks/`), figures (`results/figures/`), and journal PDF (`journal/`) into `docs/site_docs/`:

```bash
./docs/sync_docs.sh
```

---

## 2. Run Local Preview Server (`mkdocs serve`)

```bash
source venv/bin/activate
mkdocs serve -f docs/mkdocs.yml --dev-addr 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/covid19-xray-adversarial-attack/` in your web browser.

---

## 3. Build Static Production Site (`mkdocs build`)

```bash
mkdocs build -f docs/mkdocs.yml --strict -d ../site
```

The static HTML/CSS/JS site is generated in the `site/` directory.

---

## 4. Automated GitHub Actions CI/CD (`.github/workflows/docs.yml`)

On every push to the `main` or `master` branch modifying `docs/**`, `notebooks/**`, `results/figures/**`, or `journal/**`, the GitHub Actions workflow automatically:

1. Provisions an Ubuntu runner with Python `3.11`.
2. Installs documentation dependencies from `docs/requirements.txt`.
3. Synchronizes research artifacts via `./docs/sync_docs.sh`.
4. Executes `mkdocs build -f docs/mkdocs.yml --strict --verbose -d ../site` with `CI=true`.
5. Uploads the `./site` artifact and deploys it to **GitHub Pages** (`https://hisyam99.github.io/covid19-xray-adversarial-attack/`).

---

[:octicons-arrow-right-24: Continue to Conclusions & Future Work](../conclusions/index.md){ .md-button .md-button--primary }
