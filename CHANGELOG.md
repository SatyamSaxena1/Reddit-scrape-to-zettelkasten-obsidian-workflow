# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - 2025-08-27
- Initial public release
- Scraper: read CSV of saved posts (id,permalink), save JSON and media, error logging
- Processor: generate Obsidian-ready notes (wikilinks, tags, concept notes), copy media
- Graphs: export similarity/tags/concepts/syndicates, JSON + GEXF/GraphML, D3 HTML viewer
- ZIP exporter for notes, media, and graphs

Notes
- Some posts may fail due to 403/500 (removed/private/rate-limited); see output/errors.log
- Use `--graph-mode syndicates` for a sparser concepts map
