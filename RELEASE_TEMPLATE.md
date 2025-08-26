<!-- Release description template for GitHub Releases -->

## What’s new
- Initial public release of the end-to-end pipeline
- Scrape Reddit saved posts from CSV (id,permalink)
- Generate Obsidian-ready notes with wikilinks, tags, concept notes
- Export concept-centric graphs (similarity/tags/concepts/syndicates)
- Optional ZIP bundle with notes, media, graphs

## Highlights
- Obsidian-first: clean wikilinks + frontmatter
- Concept/syndicates graph for less clutter and better navigation
- Incremental-safe: re-run without duplicating work

## Getting started
1) Install dependencies (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
2) Scrape from saved_posts.csv
```powershell
python .\reddit_scraper.py --non-interactive --saved-file .\saved_posts.csv --output-dir output
```
3) Process to Obsidian (edit your vault paths)
```powershell
python .\zettel_processor.py --input-dir output --zettel-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Zettels" --media-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Attachments" --graph-mode syndicates --recompute-all
```

## Notes
- Some posts may fail with 403/500 (removed/private/rate-limited). See output/errors.log
- Use --limit-docs for a quick trial; syndicates mode for a sparser graph

## Video
- Add your YouTube video link here
