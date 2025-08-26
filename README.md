# Obsidian Reddit Zettelkasten

End-to-end pipeline to turn Reddit saved posts into Obsidian-ready Zettelkasten notes and interactive graphs.

## Quickstart (Windows PowerShell)

`powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 1) Scrape from your CSV of saved posts (id,permalink)
python .\reddit_scraper.py --non-interactive --saved-file .\saved_posts.csv --output-dir output

# 2) Process into Obsidian vault (adjust paths)
python .\zettel_processor.py --input-dir output --zettel-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Zettels" --media-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Attachments" --graph-mode syndicates --recompute-all

# 3) Optional: create a ZIP you can import elsewhere
python .\export_zip.py --notes-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Zettels" --media-dir "C:\\Path\\To\\Obsidian Vault\\Reddit Attachments" --graphs-dir output --out output\\obsidian_export.zip
`

See README for usage details.
