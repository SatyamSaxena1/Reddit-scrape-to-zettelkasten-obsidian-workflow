import argparse
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def add_dir_to_zip(zf: ZipFile, src_dir: Path, arc_base: str, patterns=None):
    if not src_dir or not src_dir.exists():
        return 0
    count = 0
    for root, _, files in os.walk(src_dir):
        for fn in files:
            p = Path(root) / fn
            if patterns:
                ok = False
                for pat in patterns:
                    if p.match(pat):
                        ok = True
                        break
                if not ok:
                    continue
            rel = p.relative_to(src_dir)
            arcname = str(Path(arc_base) / rel)
            zf.write(p, arcname, compress_type=ZIP_DEFLATED)
            count += 1
    return count


def add_files(zf: ZipFile, files: list[Path], arc_base: str):
    count = 0
    for f in files:
        if f and f.exists():
            arcname = str(Path(arc_base) / f.name)
            zf.write(f, arcname, compress_type=ZIP_DEFLATED)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Package notes, media, and graphs into a ZIP for Obsidian import")
    ap.add_argument("--notes-dir", required=True, help="Folder containing generated Markdown notes (e.g., Obsidian Vault/Reddit Zettels)")
    ap.add_argument("--media-dir", help="Folder containing copied media (e.g., Obsidian Vault/Reddit Attachments)")
    ap.add_argument("--graphs-dir", default="output", help="Folder containing graph exports (network.json, concept_graph.json/html, etc.)")
    ap.add_argument("--out", default=None, help="Output zip file path (default: output/obsidian_export.zip)")
    args = ap.parse_args()

    notes_dir = Path(args.notes_dir)
    media_dir = Path(args.media_dir) if args.media_dir else None
    graphs_dir = Path(args.graphs_dir)
    out = Path(args.out) if args.out else Path("output/obsidian_export.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        n_notes = add_dir_to_zip(zf, notes_dir, arc_base="Reddit Zettels")
        n_media = add_dir_to_zip(zf, media_dir, arc_base="Reddit Attachments") if media_dir else 0
        # Graph artifacts (top-level only)
        graph_files = [
            graphs_dir / "network.json",
            graphs_dir / "concept_graph.json",
            graphs_dir / "concept_graph.html",
            graphs_dir / "semantic_graph.gexf",
            graphs_dir / "semantic_graph.graphml",
            graphs_dir / "summary.json",
        ]
        n_graphs = add_files(zf, graph_files, arc_base="Graphs")

    print({
        "zip": str(out),
        "notes": n_notes,
        "media": n_media,
        "graphs": n_graphs,
    })


if __name__ == "__main__":
    main()
