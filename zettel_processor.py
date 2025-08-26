import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from collections import Counter

import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Simple data models
# -----------------------------

@dataclass
class Doc:
    id: str
    title: str
    permalink: str
    author: str
    score: int
    selftext: str
    comments: List[str]
    media_paths: List[str]
    created_at: str

    @property
    def text(self) -> str:
        body = (self.selftext or "").strip()
        if len(body) > 6000:
            body = body[:6000]
        # Limit to first 10 comments, each up to 500 chars
        limited_comments = [(c or "").strip()[:500] for c in (self.comments or [])[:10]]
        comments_text = "\n".join(limited_comments)
        return f"{self.title}\n{body}\n{comments_text}"


# -----------------------------
# Utilities
# -----------------------------

FILENAME_SAFE = re.compile(r"[^\w\- ]+")


def slugify(text: str, max_len: int = 80) -> str:
    s = FILENAME_SAFE.sub("", text).strip().replace(" ", "-")
    return s[:max_len] or "post"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def content_hash(doc: "Doc") -> str:
    h = hashlib.sha1()
    parts = [
        doc.title or "",
        doc.selftext or "",
        "\n".join(doc.comments or []),
        "|".join(sorted([p or "" for p in doc.media_paths] or [])),
        doc.permalink or "",
        str(doc.score or 0),
    ]
    h.update("\u241E".join(parts).encode("utf-8", errors="ignore"))  # record separator
    return h.hexdigest()


def build_backlinks_index(zettel_dir: Path):
    """Generate a backlinks index note to aid Obsidian navigation.
    Looks for [[wikilinks]] across notes and creates an index of incoming links.
    """
    try:
        link_re = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
        stems = {}
        for md in zettel_dir.rglob("*.md"):
            stems[md.stem] = md.relative_to(zettel_dir).as_posix()
        incoming: Dict[str, List[str]] = {s: [] for s in stems}
        for md in zettel_dir.rglob("*.md"):
            text = md.read_text(encoding="utf-8", errors="ignore")
            links = link_re.findall(text)
            for l in links:
                target = l.strip()
                if target in incoming and md.stem != target:
                    incoming[target].append(md.relative_to(zettel_dir).as_posix())
        lines = ["# Backlinks Index\n"]
        for target, sources in sorted(incoming.items()):
            if not sources:
                continue
            lines.append(f"\n## [[{stems.get(target, target)}|{target}]]")
            for s in sorted(set(sources)):
                lines.append(f"- [[{s}]]")
        (zettel_dir / "_backlinks_index.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def load_processed(processed_file: Path) -> Dict[str, dict]:
    if processed_file.exists():
        try:
            return json.loads(processed_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_processed(processed_file: Path, data: Dict[str, dict]):
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_doc(json_path: Path) -> Optional[Doc]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Skip graph/state-like JSONs
        if isinstance(data, dict) and ("nodes" in data or "edges" in data):
            return None
        # Require core post fields
        if not isinstance(data, dict) or (not data.get("id") and not data.get("permalink")):
            return None
        if not (data.get("title") or data.get("selftext") or data.get("comments")):
            return None
        media_paths: List[str] = []
        for m in data.get("media", []) or []:
            if isinstance(m, dict) and m.get("type") == "file" and m.get("path"):
                media_paths.append(m["path"])
        return Doc(
            id=data.get("id", json_path.stem),
            title=data.get("title") or data.get("id", "Untitled"),
            permalink=data.get("permalink", ""),
            author=data.get("author", ""),
            score=int(data.get("score") or 0),
            selftext=data.get("selftext", ""),
            comments=list(data.get("comments", []) or []),
            media_paths=media_paths,
            created_at=data.get("fetched_at") or datetime.now().isoformat(),
        )
    except Exception:
        return None


# -----------------------------
# Analysis components
# -----------------------------

def extract_semantic_tags(text: str) -> List[str]:
    t = text.lower()
    tags = set()
    # Very lightweight heuristics for context categories
    if any(k in t for k in ["how to", "tutorial", "guide", "code", "algorithm", "ml", "ai", "python", "error"]):
        tags.add("technical")
    if any(k in t for k in ["help", "how do i", "stuck", "issue", "question", "why", "what is"]):
        tags.add("problem")
    if any(k in t for k in ["solution", "fixed", "resolved", "answer", "recommendation", "tip", "trick"]):
        tags.add("solution")
    if any(k in t for k in ["love", "hate", "awesome", "terrible", "angry", "happy", "sad", "excited", "rant", "celebrate"]):
        tags.add("emotional")
    return sorted(tags)


def build_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, object]:
    # Use a lower min_df for small corpora to avoid empty matrices
    min_df = 1 if len(texts) < 10 else 2
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=min_df, max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def try_build_embeddings(texts: List[str], model_name: Optional[str] = None):
    """Optionally compute sentence embeddings; returns (embeddings_matrix or None, info_message)."""
    import importlib
    model_id = model_name or "all-MiniLM-L6-v2"
    try:
        st = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st, "SentenceTransformer")
        model = SentenceTransformer(model_id)
        # Limit overly long texts for embedding speed
        clipped = [(t or "")[:8000] for t in texts]
        embs = model.encode(clipped, show_progress_bar=False, normalize_embeddings=True)
        return embs, f"embeddings:{model_id}"
    except Exception as e:
        return None, f"embeddings_unavailable:{e}"


def top_keywords(vectorizer: TfidfVectorizer, row_vec, top_n: int = 10) -> List[str]:
    if row_vec is None:
        return []
    indices = row_vec.nonzero()[1]
    if len(indices) == 0:
        return []
    scores = row_vec.data
    pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)[: top_n * 3]
    feature_names = vectorizer.get_feature_names_out()
    # Keep unique while preserving order
    seen = set()
    kws: List[str] = []
    for idx, _ in pairs:
        term = feature_names[idx]
        if term not in seen:
            kws.append(term)
            seen.add(term)
        if len(kws) >= top_n:
            break
    return kws


def compute_sentiment(analyzer: SentimentIntensityAnalyzer, text: str) -> Dict[str, float]:
    try:
        t = (text or "").strip()
        # Truncate to avoid very long texts slowing down sentiment analysis
        if len(t) > 4000:
            t = t[:4000]
        return analyzer.polarity_scores(t)
    except Exception:
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}


def build_similarity_graph(ids: List[str], matrix, threshold: float = 0.25) -> List[Tuple[str, str, float]]:
    sims = cosine_similarity(matrix)
    edges: List[Tuple[str, str, float]] = []
    n = len(ids)
    thr = max(0.0, min(1.0, float(threshold)))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sims[i, j])
            if w >= thr:  # threshold for an edge
                edges.append((ids[i], ids[j], round(w, 4)))
    return edges


def cooccurrence_pairs(text: str, window: int = 4) -> Dict[Tuple[str, str], int]:
    # Sliding window co-occurrence (unordered pairs) to capture local topical proximity
    tokens = [t for t in re.findall(r"[A-Za-z0-9']+", text.lower()) if len(t) > 2]
    pairs: Dict[Tuple[str, str], int] = {}
    n = len(tokens)
    w = max(2, int(window))
    for i in range(n):
        end = min(n, i + w)
        for j in range(i + 1, end):
            a, b = tokens[i], tokens[j]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + 1
    return pairs


# -----------------------------
# Zettel rendering
# -----------------------------

ZETTEL_HEADER = """---
id: {id}
title: {title}
source: {permalink}
author: {author}
created_at: {created_at}
tags: {front_tags}
neural:
  keywords: {keywords}
  semantic_tags: {semantic_tags}
  neural_weight: {neural_weight}
  sentiment: {sentiment}
---
"""


def render_note(doc: Doc, keywords: List[str], tags: List[str], weight: float, sentiment: Dict[str, float], related: Optional[List[Dict]] = None, front_tags: Optional[List[str]] = None, media_links: Optional[List[str]] = None, embed_media: bool = False, concept_links: Optional[List[Tuple[str, str]]] = None) -> str:
    header = ZETTEL_HEADER.format(
        id=doc.id,
        title=doc.title.replace("\n", " ").strip(),
        permalink=doc.permalink,
        author=doc.author,
        created_at=doc.created_at,
    front_tags=json.dumps(front_tags or [], ensure_ascii=False),
        keywords=json.dumps(keywords, ensure_ascii=False),
        semantic_tags=json.dumps(tags, ensure_ascii=False),
        neural_weight=round(weight, 4),
        sentiment=json.dumps(sentiment, ensure_ascii=False),
    )
    media_section = ""
    if media_links:
        if embed_media:
            # Obsidian embeds: use ![[file]] for images, [[file]] for others
            lines = []
            for p in media_links:
                ext = Path(p).suffix.lower()
                is_img = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                lines.append(("![[" if is_img else "[[") + f"{p}]]")
            media_section = "\n## Attachments\n\n" + "\n".join(lines) + "\n"
        else:
            media_lines = [f"- {p}" for p in media_links]
            media_section = "\n## Attachments\n" + "\n".join(media_lines) + "\n"

    body = f"\n## Content\n\n{doc.selftext or ''}\n\n## Top Comments\n\n" + "\n\n---\n\n".join(
        (c or "").strip()[:1500] for c in (doc.comments[:5] or [])
    )
    concepts_section = ""
    if concept_links:
        # concept_links: list of (target_path_without_ext, label)
        lines = [f"- [[{p}|{lbl}]]" for p, lbl in concept_links]
        concepts_section = "\n## Concepts\n\n" + "\n".join(lines) + "\n"

    related_section = ""
    if related:
        lines = []
        for r in related:
            title = r.get("title") or r.get("id")
            link = r.get("file")
            w = r.get("weight")
            if link:
                # Obsidian-style wikilink with alias falls back to standard markdown link if not in Obsidian
                link_stem = Path(link).stem
                lines.append(f"- [[{link_stem}|{title}]] (w={w:.2f})")
            else:
                lines.append(f"- {title} (id={r.get('id')}, w={w:.2f})")
        related_section = "\n## Related\n\n" + "\n".join(lines) + "\n"
    return header + media_section + body + concepts_section + related_section + "\n"


# -----------------------------
# Main pipeline
# -----------------------------

def process_once(input_dir: Path, zettel_dir: Path, state_file: Path, graph_file: Path,
                 similarity_threshold: float = 0.25,
                 co_window: int = 4,
                 top_related: int = 5,
                 recompute_all: bool = False,
                 export_gexf: bool = True,
                 export_graphml: bool = True,
                 use_embeddings: bool = False,
                 embeddings_model: Optional[str] = None,
                 media_dir: Optional[Path] = None,
                 embed_media: bool = True,
                 export_concept_graph: bool = True,
                 concept_top_posts: int = 3,
                 only_ids: Optional[List[str]] = None,
                 limit_docs: Optional[int] = None,
                 graph_mode: str = "similarity",
                 tag_edge_min: int = 1) -> Dict[str, int]:
    processed = load_processed(state_file)
    # Prime processed from existing zettel filenames to avoid duplicates
    try:
        for md in zettel_dir.rglob("*.md"):
            name = md.name
            # Expect: TIMESTAMP-ID-SLUG.md
            parts = name.split("-")
            if len(parts) >= 3 and parts[0].isdigit():
                doc_id = parts[1]
                rel = md.relative_to(zettel_dir).as_posix()
                processed.setdefault(doc_id, {"file": rel})
    except Exception:
        pass
    analyzer = SentimentIntensityAnalyzer()

    # Collect docs
    json_files = sorted(input_dir.glob("*.json"))
    docs: List[Doc] = []
    for p in json_files:
        # Skip known non-post JSONs
        if p.resolve() == state_file.resolve() or p.resolve() == graph_file.resolve():
            continue
        doc = read_doc(p)
        if not doc:
            continue
        docs.append(doc)

    # Optional filtering
    if only_ids:
        idset = set(only_ids)
        docs = [d for d in docs if d.id in idset]
    if limit_docs and isinstance(limit_docs, int) and limit_docs > 0:
        docs = docs[:limit_docs]

    if not docs:
        return {"seen": 0, "new": 0}

    # Build representations across all docs
    all_texts = [d.text for d in docs]
    # Always build TF-IDF for keyword extraction
    vectorizer, tfidf_matrix = build_vectorizer(all_texts)
    # Optionally build embeddings for similarity
    emb_matrix = None
    rep_info = "tfidf"
    if use_embeddings:
        emb_matrix, rep_info = try_build_embeddings(all_texts, embeddings_model)
    matrix = emb_matrix if emb_matrix is not None else tfidf_matrix

    # Precompute top keywords for all docs (used for notes and concept graph)
    doc_keywords: Dict[str, List[str]] = {}
    all_kw_set: set = set()
    for idx, d in enumerate(docs):
        row_vec = tfidf_matrix[idx]
        kws = top_keywords(vectorizer, row_vec, top_n=10)
        doc_keywords[d.id] = kws
        all_kw_set.update(kws)
    # Precompute semantic tags per doc (for tags-based graph mode)
    doc_tags: Dict[str, List[str]] = {d.id: extract_semantic_tags(d.text) for d in docs}
    # Predeclare concept note paths for wikilinks in notes
    concept_paths: Dict[str, str] = {k: f"Concepts/{k}" for k in sorted(all_kw_set)}
    id_list = [d.id for d in docs]

    # Similarity graph
    sim_edges = build_similarity_graph(id_list, matrix, threshold=similarity_threshold)

    # Build NetworkX graph to compute metrics
    G = nx.Graph()
    for d in docs:
        G.add_node(d.id, title=d.title, score=d.score)
    for s, t, w in sim_edges:
        G.add_edge(s, t, weight=w)

    # Metrics (normalized)
    try:
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, normalized=True, k=None)
        clustering = nx.clustering(G, weight="weight")
        pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes}
    except Exception:
        centrality, betweenness, clustering, pagerank = {}, {}, {}, {}

    # Build neighbor map for top related suggestions
    neighbor_map: Dict[str, List[Tuple[str, float]]] = {i: [] for i in id_list}
    for s, t, w in sim_edges:
        neighbor_map[s].append((t, w))
        neighbor_map[t].append((s, w))
    for k in neighbor_map:
        neighbor_map[k].sort(key=lambda x: x[1], reverse=True)

    # Community detection (greedy modularity)
    communities: Dict[str, int] = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight")) if G.number_of_edges() > 0 else []
        for idx_c, nodes in enumerate(comms):
            for n in nodes:
                communities[str(n)] = idx_c
    except Exception:
        pass

    # Process docs: create new notes or update if content changed or recompute_all
    new_count = 0
    updated_count = 0
    zettel_dir.mkdir(parents=True, exist_ok=True)
    for idx, d in enumerate(docs):
        h = content_hash(d)
        prev = processed.get(d.id)
        needs_content = recompute_all or prev is None or prev.get("hash") != h

        if needs_content:
            # Use precomputed TF-IDF keywords
            kws = doc_keywords.get(d.id, [])
            tags = extract_semantic_tags(d.text)
            sent = compute_sentiment(analyzer, d.text)
            # Neural weight: combine Reddit score (capped) and centrality/betweenness signals
            score_norm = min(max(d.score, 0), 2000) / 2000.0
            c = centrality.get(d.id, 0.0)
            b = betweenness.get(d.id, 0.0)
            pr = pagerank.get(d.id, 0.0)
            weight = 0.5 * score_norm + 0.25 * c + 0.15 * b + 0.10 * pr

            # Build related list (top K)
            nbrs = neighbor_map.get(d.id, [])[: max(0, int(top_related))]
            id_to_title = {x.id: x.title for x in docs}
            related = []
            for nid, nw in nbrs:
                rel = {"id": nid, "title": id_to_title.get(nid, nid), "weight": nw}
                if processed.get(nid, {}).get("file"):
                    rel["file"] = processed[nid]["file"]
                related.append(rel)

            # Copy media into target media_dir (if provided) and build Obsidian-friendly paths
            obsidian_media_links: List[str] = []
            if media_dir:
                try:
                    media_dir.mkdir(parents=True, exist_ok=True)
                    for src in (d.media_paths or []):
                        sp = Path(src)
                        if sp.exists():
                            dst = media_dir / sp.name
                            if str(sp.resolve()) != str(dst.resolve()):
                                try:
                                    # Use binary-safe copy
                                    with open(sp, 'rb') as r, open(dst, 'wb') as w:
                                        w.write(r.read())
                                except Exception:
                                    pass
                            obsidian_media_links.append(dst.name)
                        else:
                            # If missing, keep original string for visibility
                            obsidian_media_links.append(sp.name)
                except Exception:
                    pass
            else:
                # Fallback: keep original relative paths in list form
                obsidian_media_links = [Path(p).name for p in (d.media_paths or [])]

            # Determine note placement (community folders optional)
            comm = communities.get(d.id)
            folder = f"community-{int(comm):02d}" if (isinstance(comm, int) and os.environ.get("ZETTEL_USE_COMMUNITY_FOLDERS") == "1") else ""
            # Determine base filename (preserve existing basename if present)
            prior_file = prev.get("file") if prev else None
            base_name = (Path(prior_file).name if prior_file else f"{now_stamp()}-{d.id}-{slugify(d.title)}.md")
            rel_path = (Path(folder) / base_name) if folder else Path(base_name)
            note_path = zettel_dir / rel_path
            note_path.parent.mkdir(parents=True, exist_ok=True)
            # If moving from old path, remove the old file to avoid duplicates
            try:
                if prior_file:
                    old_path = zettel_dir / prior_file
                    if old_path.exists() and old_path.resolve() != note_path.resolve():
                        old_path.unlink(missing_ok=True)
            except Exception:
                pass
            front_tags = sorted(set(["reddit", "zettel"]) | set(tags))
            # Build concept wikilinks for this doc
            concept_links = []
            for k in doc_keywords.get(d.id, []):
                p = concept_paths.get(k)
                if p:
                    concept_links.append((p, k))

            note_path.write_text(
                render_note(
                    d, kws, tags, weight, sent,
                    related=related,
                    front_tags=front_tags,
                    media_links=obsidian_media_links,
                    embed_media=embed_media,
                    concept_links=concept_links,
                ),
                encoding="utf-8",
            )

            entry = {
                "title": d.title,
                "keywords": kws,
                "tags": tags,
                "sentiment": sent,
                "weight": weight,
                "centrality": centrality.get(d.id, 0.0),
                "betweenness": betweenness.get(d.id, 0.0),
                "pagerank": pagerank.get(d.id, 0.0),
                "clustering": clustering.get(d.id, 0.0),
                "created_at": d.created_at,
                "hash": h,
                "file": rel_path.as_posix(),
                "community": communities.get(d.id),
            }
            processed[d.id] = entry
            if prev is None:
                new_count += 1
            else:
                updated_count += 1

    # Refresh metrics and weights for all docs to avoid stale state
    for d in docs:
        score_norm = min(max(d.score, 0), 2000) / 2000.0
        c = centrality.get(d.id, 0.0)
        b = betweenness.get(d.id, 0.0)
        pr = pagerank.get(d.id, 0.0)
        weight = 0.5 * score_norm + 0.25 * c + 0.15 * b + 0.10 * pr
        entry = processed.setdefault(d.id, {"title": d.title, "created_at": d.created_at})
        entry.update({
            "weight": weight,
            "centrality": c,
            "betweenness": b,
            "pagerank": pr,
            "clustering": clustering.get(d.id, 0.0),
            "community": communities.get(d.id),
        })

    # Persist processed state
    save_processed(state_file, processed)

    # Persist graph (nodes + edges)
    # Build a simple word co-occurrence network across all documents
    co_counts: Dict[Tuple[str, str], int] = {}
    for d in docs:
        pairs = cooccurrence_pairs(d.text, window=co_window)
        for k, v in pairs.items():
            co_counts[k] = co_counts.get(k, 0) + v

    # Build nodes list (optionally filter to tagged docs in tag mode)
    GENERIC_TAGS = {"reddit", "zettel"}
    node_docs = [d for d in docs if any(t for t in doc_tags.get(d.id, []) if t not in GENERIC_TAGS)] if graph_mode == "tags" else docs

    # Build edges for export
    if graph_mode == "tags":
        # Connect posts that share at least tag_edge_min semantic tags (excluding generic)
        tag_min = max(1, int(tag_edge_min))
        edges_export: List[Tuple[str, str, float]] = []
        for i in range(len(node_docs)):
            di = node_docs[i]
            ti = set(t for t in doc_tags.get(di.id, []) if t not in GENERIC_TAGS)
            if not ti:
                continue
            for j in range(i + 1, len(node_docs)):
                dj = node_docs[j]
                tj = set(t for t in doc_tags.get(dj.id, []) if t not in GENERIC_TAGS)
                if not tj:
                    continue
                inter = ti & tj
                if len(inter) >= tag_min:
                    edges_export.append((di.id, dj.id, float(len(inter))))
    else:
        edges_export = sim_edges

    if graph_mode == "concepts":
        # Build concept-only graph: keywords as nodes; edges by co-occurrence across posts
        df_c: Dict[str, int] = {}
        co_c: Dict[Tuple[str, str], int] = {}
        for d in docs:
            kws = sorted(set(doc_keywords.get(d.id, [])))
            for k in kws:
                df_c[k] = df_c.get(k, 0) + 1
            for i in range(len(kws)):
                for j in range(i + 1, len(kws)):
                    a, b = kws[i], kws[j]
                    key = (a, b) if a < b else (b, a)
                    co_c[key] = co_c.get(key, 0) + 1
        graph = {
            "nodes": [
                {"id": k, "title": k, "df": df_c.get(k, 0), "weight": float(df_c.get(k, 0))}
                for k in sorted(df_c.keys())
            ],
            "edges": [
                {"source": a, "target": b, "weight": float(w)} for (a, b), w in sorted(co_c.items(), key=lambda x: x[1], reverse=True)
            ],
            "mode": graph_mode,
        }
    elif graph_mode == "syndicates":
        # Cluster concepts into communities (syndicates) and aggregate edges between them
        df_c: Dict[str, int] = {}
        co_c: Dict[Tuple[str, str], int] = {}
        kw_docs_map: Dict[str, set] = {}
        for d in docs:
            kws = sorted(set(doc_keywords.get(d.id, [])))
            for k in kws:
                df_c[k] = df_c.get(k, 0) + 1
                kw_docs_map.setdefault(k, set()).add(d.id)
            for i in range(len(kws)):
                for j in range(i + 1, len(kws)):
                    a, b = kws[i], kws[j]
                    key = (a, b) if a < b else (b, a)
                    co_c[key] = co_c.get(key, 0) + 1
        # Build concept graph
        CG = nx.Graph()
        for k, freq in df_c.items():
            CG.add_node(k, df=freq)
        for (a, b), w in co_c.items():
            CG.add_edge(a, b, weight=float(w))
        # Communities
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(CG, weight="weight")) if CG.number_of_edges() > 0 else []
        except Exception:
            comms = []
        if not comms:
            # Fallback: one syndicate with all concepts
            comms = [set(CG.nodes())]
        # Map concept -> syndicate id
        concept_to_syn: Dict[str, int] = {}
        for i, nodes in enumerate(comms):
            for n in nodes:
                concept_to_syn[n] = i
        # Build syndicate nodes summary with descriptive labels
        STOP = {
            "the","a","an","and","or","for","to","of","in","on","with","by","from","at",
            "is","are","was","were","be","as","it","that","this","these","those","how","what","why"
        }
        syn_nodes = []
        for i, nodes in enumerate(comms):
            # Top terms by df
            tops = sorted(list(nodes), key=lambda k: df_c.get(k, 0), reverse=True)[:5]
            # Derive phrase label from member posts' titles (frequent bigrams)
            # Collect member docs
            member_doc_ids: set = set()
            for k in nodes:
                member_doc_ids |= kw_docs_map.get(k, set())
            titles = [d.title for d in docs if d.id in member_doc_ids]
            # Tokenize and count bigrams excluding stopwords and very short tokens
            tokens = []
            for t in titles:
                toks = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9+_-]+", t or "") if len(w) > 2]
                tokens.append([w for w in toks if w not in STOP])
            bigram_counts = Counter()
            for seq in tokens:
                for j in range(len(seq)-1):
                    a, b = seq[j], seq[j+1]
                    if a in STOP or b in STOP:
                        continue
                    bigram_counts[(a, b)] += 1
            best_phrase = None
            if bigram_counts:
                ((a, b), cnt) = max(bigram_counts.items(), key=lambda x: x[1])
                if cnt >= 2:
                    best_phrase = f"{a} {b}"
            # Compose label
            label = (best_phrase.title() if best_phrase else ", ".join(tops[:3]).title()) if tops else f"Syndicate {i}"
            size = len(nodes)
            df_sum = sum(df_c.get(k, 0) for k in nodes)
            syn_nodes.append({
                "id": f"S{i}",
                "title": label,
                "size": size,
                "df_sum": float(df_sum),
                "weight": float(df_sum),
            })
        # Aggregate inter-syndicate edges
        syn_edges_acc: Dict[Tuple[int, int], float] = {}
        for (a, b), w in co_c.items():
            ia, ib = concept_to_syn.get(a), concept_to_syn.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            key = (ia, ib) if ia < ib else (ib, ia)
            syn_edges_acc[key] = syn_edges_acc.get(key, 0.0) + float(w)
        syn_edges = [
            {"source": f"S{ia}", "target": f"S{ib}", "weight": wt}
            for (ia, ib), wt in sorted(syn_edges_acc.items(), key=lambda x: x[1], reverse=True)
        ]
        graph = {
            "nodes": syn_nodes,
            "edges": syn_edges,
            "mode": graph_mode,
        }
    else:
        graph = {
            "nodes": [
                {
                    "id": d.id,
                    "title": d.title,
                    "score": d.score,
                    # Use live metrics when available
                    "weight": round(0.5 * min(max(d.score, 0), 2000) / 2000.0
                               + 0.25 * centrality.get(d.id, 0.0)
                               + 0.15 * betweenness.get(d.id, 0.0)
                               + 0.10 * pagerank.get(d.id, 0.0), 4),
                    "tags": processed.get(d.id, {}).get("tags", []),
                    "centrality": centrality.get(d.id, 0.0),
                    "betweenness": betweenness.get(d.id, 0.0),
                    "pagerank": pagerank.get(d.id, 0.0),
                    "clustering": clustering.get(d.id, 0.0),
                    "community": communities.get(d.id),
                }
                for d in node_docs
            ],
            "edges": [
                {"source": s, "target": t, "weight": w} for s, t, w in edges_export
            ],
            "cooccurrence": [
                {"a": a, "b": b, "count": c} for (a, b), c in sorted(co_counts.items(), key=lambda x: x[1], reverse=True)[:5000]
            ],
            "mode": graph_mode,
        }
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build and export a concept-only graph (keywords as nodes)
    concept_paths: Dict[str, str] = {}
    if export_concept_graph:
        # Document frequency per keyword and co-occurrence per doc
        df: Dict[str, int] = {}
        co: Dict[Tuple[str, str], int] = {}
        kw_docs: Dict[str, List[str]] = {}
        for d in docs:
            kws = sorted(set(doc_keywords.get(d.id, [])))
            for k in kws:
                df[k] = df.get(k, 0) + 1
                kw_docs.setdefault(k, []).append(d.id)
            # pairwise
            for i in range(len(kws)):
                for j in range(i + 1, len(kws)):
                    a, b = kws[i], kws[j]
                    key = (a, b) if a < b else (b, a)
                    co[key] = co.get(key, 0) + 1

        # Optional: sample top posts per keyword for context (by PageRank of nodes' posts)
        pr_default = {d.id: pagerank.get(d.id, 0.0) for d in docs}
        kw_examples: Dict[str, List[str]] = {}
        for k, ids in kw_docs.items():
            top = sorted(ids, key=lambda nid: pr_default.get(nid, 0.0), reverse=True)[: max(0, int(concept_top_posts))]
            kw_examples[k] = top

        concept_graph = {
            "nodes": [
                {"id": k, "df": df[k], "examples": kw_examples.get(k, [])}
                for k in sorted(df.keys())
            ],
            "edges": [
                {"source": a, "target": b, "weight": w}
                for (a, b), w in sorted(co.items(), key=lambda x: x[1], reverse=True)
            ],
        }
        (graph_file.parent / "concept_graph.json").write_text(json.dumps(concept_graph, ensure_ascii=False, indent=2), encoding="utf-8")

        # Create Concept notes in vault for Obsidian navigation
        try:
            concepts_dir = zettel_dir / "Concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            # Build concept neighbors map
            c_neighbors: Dict[str, List[Tuple[str, int]]] = {}
            for (a, b), w in co.items():
                c_neighbors.setdefault(a, []).append((b, w))
                c_neighbors.setdefault(b, []).append((a, w))
            for k, doc_ids in kw_docs.items():
                neighbors = sorted(c_neighbors.get(k, []), key=lambda x: x[1], reverse=True)[:15]
                lines = [f"---\ntitle: {k}\ntype: concept\ndf: {df.get(k,0)}\n---\n"]
                if neighbors:
                    lines.append("## Related Concepts\n")
                    for n, w in neighbors:
                        lines.append(f"- [[Concepts/{n}|{n}]] (w={w})")
                lines.append("\n## Posts\n")
                for nid in sorted(doc_ids, key=lambda nid: pagerank.get(nid, 0.0), reverse=True)[:50]:
                    p = processed.get(nid, {})
                    if p.get("file"):
                        stem = Path(p["file"]).stem
                        lines.append(f"- [[{stem}|{docs[id_list.index(nid)].title}]]")
                c_path = (concepts_dir / f"{k}.md")
                c_path.write_text("\n".join(lines), encoding="utf-8")
                concept_paths[k] = f"Concepts/{k}"
        except Exception:
            pass

        # Simple D3 HTML for concept graph
        try:
            html_c = """
<!doctype html>
<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>Concept Graph</title>\n<style>
body { margin: 0; font-family: system-ui, Segoe UI, Arial; }
#legend { position: fixed; top: 8px; left: 8px; background: rgba(255,255,255,.9); padding: 8px 10px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.15); font-size: 12px; }
svg { width: 100vw; height: 100vh; display: block; background: #0b1020; }
.node circle { stroke: #fff; stroke-width: .5px; }
.node text { pointer-events: none; font-size: 11px; fill: #eee; text-shadow: 0 1px 2px #000; }
.link { stroke: rgba(200,200,220,.3); stroke-width: 1px; }
</style>\n</head>\n<body>
<div id=\"legend\">Keywords as nodes • Edge weight = co-post count</div>
<svg></svg>
<script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
<script>
fetch('concept_graph.json')
    .then(r => r.json())
    .then(data => {
        const svg = d3.select('svg');
        const width = window.innerWidth, height = window.innerHeight;
        const nodes = data.nodes.map(d => Object.assign({}, d));
        const links = data.edges.map(d => Object.assign({}, d));
        const dfExtent = d3.extent(nodes, d => d.df || 1);
        const size = d3.scaleSqrt().domain(dfExtent).range([4, 18]);
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(d => 120 - 10*(d.weight||0)).strength(0.15))
            .force('charge', d3.forceManyBody().strength(-45))
            .force('center', d3.forceCenter(width/2, height/2))
            .force('collide', d3.forceCollide().radius(d => size(d.df||1) + 3));
        const g = svg.append('g');
        const link = g.append('g').attr('stroke-opacity', 0.6)
            .selectAll('line').data(links).join('line').attr('class','link');
        const node = g.append('g').selectAll('g').data(nodes).join('g').attr('class','node');
        node.append('circle').attr('r', d => size(d.df||1)).attr('fill', '#5ab0f2');
        node.append('title').text(d => `${d.id} (df=${d.df})`);
        node.append('text').text(d => d.id).attr('x', 8).attr('y', 3);
        node.call(d3.drag()
            .on('start', (event,d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
            .on('drag', (event,d) => { d.fx=event.x; d.fy=event.y; })
            .on('end', (event,d) => { if (!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));
        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });
        const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e)=>g.attr('transform', e.transform));
        svg.call(zoom);
    });
</script>
</body>\n</html>
"""
            (graph_file.parent / "concept_graph.html").write_text(html_c, encoding="utf-8")
        except Exception:
            pass

    # Export to GEXF / GraphML for external tools
    try:
        # enrich node attributes before export
        for d in docs:
            n = d.id
            if n in G:
                G.nodes[n]["centrality"] = centrality.get(n, 0.0)
                G.nodes[n]["betweenness"] = betweenness.get(n, 0.0)
                G.nodes[n]["pagerank"] = pagerank.get(n, 0.0)
                G.nodes[n]["clustering"] = clustering.get(n, 0.0)
                G.nodes[n]["community"] = communities.get(n)
                G.nodes[n]["rep"] = rep_info
        if export_gexf:
            nx.write_gexf(G, (graph_file.parent / "semantic_graph.gexf").as_posix())
        if export_graphml:
            nx.write_graphml(G, (graph_file.parent / "semantic_graph.graphml").as_posix())
    except Exception:
        pass

    # Write a compact community summary for quick browsing
    try:
        comm_to_nodes: Dict[int, List[str]] = {}
        for nid, cidx in communities.items():
            if cidx is None:
                continue
            comm_to_nodes.setdefault(int(cidx), []).append(nid)
        lines = ["# Community summary\n"]
        for cidx, node_ids in sorted(comm_to_nodes.items()):
            subset = [d for d in docs if d.id in node_ids]
            # top 10 nodes by PageRank
            top_nodes = sorted(subset, key=lambda x: pagerank.get(x.id, 0.0), reverse=True)[:10]
            lines.append(f"\n## Community {cidx} ({len(node_ids)} nodes)")
            for tn in top_nodes:
                lines.append(f"- {tn.title} (id={tn.id}, pr={pagerank.get(tn.id, 0.0):.4f})")
        (graph_file.parent / "community_summary.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    # Write a simple D3 viewer HTML for the graph JSON
    try:
        html = """
<!doctype html>
<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>Zettel Graph</title>\n<style>
body {{ margin: 0; font-family: system-ui, Segoe UI, Arial; }}
#legend {{ position: fixed; top: 8px; left: 8px; background: rgba(255,255,255,.9); padding: 8px 10px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.15); font-size: 12px; }}
svg {{ width: 100vw; height: 100vh; display: block; background: #0b1020; }}
.node circle {{ stroke: #fff; stroke-width: .5px; }}
.node text {{ pointer-events: none; font-size: 10px; fill: #eee; text-shadow: 0 1px 2px #000; }}
.link {{ stroke: rgba(200,200,220,.3); stroke-width: 1px; }}
</style>\n</head>\n<body>
<div id=\"legend\">Left-drag: move • Scroll: zoom • Color=community • Size=weight</div>
<svg></svg>
<script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
<script>
fetch('{graph_file.name}')
    .then(r => r.json())
    .then(data => {
        const svg = d3.select('svg');
        const width = window.innerWidth, height = window.innerHeight;
        const color = d3.scaleOrdinal(d3.schemeTableau10);
        const nodes = data.nodes.map(d => Object.assign({}, d));
        const links = data.edges.map(d => Object.assign({}, d));
        const weightExtent = d3.extent(nodes, d => d.weight ?? 0);
        const size = d3.scaleLinear().domain(weightExtent).range([4, 14]);

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(d => 60 - 40*(d.weight||0)).strength(0.2))
            .force('charge', d3.forceManyBody().strength(-30))
            .force('center', d3.forceCenter(width/2, height/2))
            .force('collide', d3.forceCollide().radius(d => size(d.weight||0) + 2));

        const g = svg.append('g');
        const link = g.append('g').attr('stroke-opacity', 0.6)
            .selectAll('line').data(links).join('line').attr('class','link');
        const node = g.append('g').selectAll('g').data(nodes).join('g').attr('class','node');
        node.append('circle').attr('r', d => size(d.weight||0)).attr('fill', d => color(d.community ?? 0));
        node.append('title').text(d => `${d.title}`);
        node.append('text').text(d => d.title).attr('x', 8).attr('y', 3);

        node.call(d3.drag()
            .on('start', (event,d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
            .on('drag', (event,d) => { d.fx=event.x; d.fy=event.y; })
            .on('end', (event,d) => { if (!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));

        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e)=>g.attr('transform', e.transform));
        svg.call(zoom);
    });
</script>
</body>\n</html>
"""
        (graph_file.parent / "graph.html").write_text(html.replace("__GRAPH_JSON__", graph_file.name), encoding="utf-8")
    except Exception:
        pass

    # Remove backlinks index to avoid graph clutter
    try:
        idx = zettel_dir / "_backlinks_index.md"
        if idx.exists():
            idx.unlink()
    except Exception:
        pass

    return {"seen": len(docs), "new": new_count, "updated": updated_count}


def watch(input_dir: Path, zettel_dir: Path, state_file: Path, graph_file: Path,
          interval: float = 10.0,
          similarity_threshold: float = 0.25,
          co_window: int = 4,
          top_related: int = 5,
          recompute_all: bool = False,
          use_embeddings: bool = False,
          embeddings_model: Optional[str] = None,
          media_dir: Optional[Path] = None,
          embed_media: bool = True):
    last_counts = (0, 0)
    while True:
        try:
            stats = process_once(input_dir, zettel_dir, state_file, graph_file,
                                 similarity_threshold=similarity_threshold,
                                 co_window=co_window,
                                 top_related=top_related,
                                 recompute_all=recompute_all,
                                 use_embeddings=use_embeddings,
                                 embeddings_model=embeddings_model,
                                 media_dir=media_dir,
                                 embed_media=embed_media)
            # Only print when there is a change to keep logs quiet
            if (stats["seen"], stats["new"]) != last_counts:
                print(f"[zettel] seen={stats['seen']} new={stats['new']} updated={stats.get('updated', 0)} @ {datetime.now().isoformat(timespec='seconds')}")
                last_counts = (stats["seen"], stats["new"]) 
        except KeyboardInterrupt:
            print("Stopping watcher...")
            return
        except Exception as e:
            print(f"[zettel] error: {e}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Zettelkasten processor for Reddit scraper outputs")
    parser.add_argument("--input-dir", default="output", help="Directory containing per-post JSON files")
    parser.add_argument("--zettel-dir", default="zettel", help="Directory to write Markdown notes")
    parser.add_argument("--state-file", default="output/processed_ids.json", help="Path to processed state JSON")
    parser.add_argument("--graph-file", default="output/network.json", help="Where to write the network graph JSON")
    parser.add_argument("--interval", type=float, default=10.0, help="Polling interval when watching (seconds)")
    parser.add_argument("--watch", action="store_true", help="Watch input directory and process incrementally")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="Cosine similarity threshold for edges [0-1]")
    parser.add_argument("--co-window", type=int, default=4, help="Sliding window for word co-occurrence")
    parser.add_argument("--top-related", type=int, default=5, help="Number of related notes to include per note")
    parser.add_argument("--recompute-all", action="store_true", help="Rewrite all notes to refresh keywords/tags/related")
    parser.add_argument("--use-embeddings", action="store_true", help="Use sentence-transformers embeddings for similarity edges")
    parser.add_argument("--embeddings-model", default=None, help="Embeddings model id (e.g., all-MiniLM-L6-v2)")
    parser.add_argument("--media-dir", default=None, help="Directory to copy media into (e.g., Obsidian vault attachments)")
    parser.add_argument("--no-embed-media", action="store_true", help="List media filenames instead of embedding in notes")
    parser.add_argument("--limit-docs", type=int, default=0, help="Process only the first N docs (after filtering)")
    parser.add_argument("--only-ids", default=None, help="Comma-separated list of post IDs to process")
    parser.add_argument("--use-community-folders", action="store_true", help="Place notes into per-community subfolders (disabled by default)")
    parser.add_argument("--graph-mode", default="similarity", choices=["similarity", "tags", "concepts", "syndicates"], help="Graph edges based on similarity/tags, or show concepts-only or concept syndicates")
    parser.add_argument("--tag-edge-min", type=int, default=1, help="Minimum shared tags to connect two posts when graph-mode=tags")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    zettel_dir = Path(args.zettel_dir)
    state_file = Path(args.state_file)
    graph_file = Path(args.graph_file)

    if args.use_community_folders:
        os.environ["ZETTEL_USE_COMMUNITY_FOLDERS"] = "1"
    else:
        os.environ.pop("ZETTEL_USE_COMMUNITY_FOLDERS", None)

    if args.watch:
        watch(input_dir, zettel_dir, state_file, graph_file,
              interval=args.interval,
              similarity_threshold=args.similarity_threshold,
              co_window=args.co_window,
              top_related=args.top_related,
              recompute_all=args.recompute_all,
              use_embeddings=args.use_embeddings,
              embeddings_model=args.embeddings_model,
              media_dir=Path(args.media_dir) if args.media_dir else None,
              embed_media=not args.no_embed_media)
    else:
        only_ids = [s.strip() for s in args.only_ids.split(",")] if args.only_ids else None
        stats = process_once(input_dir, zettel_dir, state_file, graph_file,
                             similarity_threshold=args.similarity_threshold,
                             co_window=args.co_window,
                             top_related=args.top_related,
                             recompute_all=args.recompute_all,
                             use_embeddings=args.use_embeddings,
                             embeddings_model=args.embeddings_model,
                             media_dir=Path(args.media_dir) if args.media_dir else None,
                             embed_media=not args.no_embed_media,
                             only_ids=only_ids,
                             limit_docs=(args.limit_docs if args.limit_docs and args.limit_docs > 0 else None),
                             graph_mode=args.graph_mode,
                             tag_edge_min=args.tag_edge_min)
        print(json.dumps({"status": "ok", **stats}))


if __name__ == "__main__":
    main()
