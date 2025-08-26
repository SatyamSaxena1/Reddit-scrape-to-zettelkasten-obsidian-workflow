[![Release](https://img.shields.io/github/v/release/SatyamSaxena1/Reddit-scrape-to-zettelkasten-obsidian-workflow?logo=github)](https://github.com/SatyamSaxena1/Reddit-scrape-to-zettelkasten-obsidian-workflow/releases/latest)

# Obsidian Reddit Zettelkasten (Release)

Turn a CSV of Reddit saved posts into Obsidian-ready Zettelkasten notes with concept graphs.

## Demo

- Desktop (GIF)
  
  ![Desktop demo](assets/PC.gif)

- iOS (video)
  
  See Release assets or YouTube link in the Release notes.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 1) Scrape your saved posts
Provide a CSV with two columns: `id,permalink`.

```powershell
python .\reddit_scraper.py --non-interactive --saved-file .\saved_posts.csv --output-dir output
```

Notes
- Some posts may return 403/500 (removed/private/rate-limited). See `output/errors.log`.
- Safe to re-run; existing files are skipped.

## 2) Generate Obsidian notes and graphs
Set your vault folders for notes and attachments, then run:

```powershell
python .\zettel_processor.py --input-dir output --zettel-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Zettels" --media-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Attachments" --graph-mode syndicates --recompute-all
```

Outputs
- Markdown notes (+ concept notes) into your vault
- Graphs in `output/` (network.json, concept_graph.json/html, GEXF/GraphML)

Useful flags
- `--limit-docs N` process a subset
- `--graph-mode [similarity|tags|concepts|syndicates]`
- `--watch` reprocess on changes

## 3) Optional: export a ZIP bundle
Bundle notes, attachments, and graphs for easy sharing/import.

```powershell
python .\export_zip.py --notes-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Zettels" --media-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Attachments" --graphs-dir output --out output\obsidian_export.zip
```

## License
See `LICENSE`.
