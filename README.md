# Obsidian Reddit Zettelkasten (Release)

Turn the chaos of saved Reddit posts into a battle-ready Obsidian Zettelkasten. Harvest ideas, forge concept links, and deploy a clean graph — like a fresh Warbond drop for your note-taking arsenal.

## Demo

- Desktop (GIF)
  
  ![Desktop demo](assets/PC.gif)

- iOS (video)
  
  See the v0.1.0 Release assets (ios.mov).

## Release
- Notes: https://github.com/SatyamSaxena1/Reddit-scrape-to-zettelkasten-obsidian-workflow/releases/tag/v0.1.0
- iOS demo (direct download): https://github.com/SatyamSaxena1/Reddit-scrape-to-zettelkasten-obsidian-workflow/releases/download/v0.1.0/ios.mov

## Why this is different
- Concept-first links: not just post-to-post — we build concept notes and cluster them into “syndicates” for high-signal maps.
- Obsidian-native: wikilinks, frontmatter tags, clean filenames. Drop-in and it just works.
- Graphs for humans: multiple modes (similarity/tags/concepts/syndicates), with a sparser default for speed.

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

Credentials/OAuth
- Create an app at https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
- Optional: REDDIT_USERNAME, REDDIT_PASSWORD for private/age-gated content or use `--oauth`
- Full guide: docs/REDDIT_OAUTH.md

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
