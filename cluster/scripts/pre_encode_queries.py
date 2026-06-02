"""
pre_encode_queries.py — STELLAR RAG Engine
===========================================
Pre-encodes RAG queries in batch for all stars in the STELLAR corpus.

Separates encode() from retrieval into two distinct phases:

  Phase 0 — Pre-encoding (this script, ~30s, run once from login node)
      Reads stellar_corpus_v7.json, builds the hybrid query for each star
      using _build_query() from engine.py, and encodes all queries in batch.
      Saves the resulting matrix to rag_cache/.

  Phase 1 — Inference (engine.py)
      retrieve() performs an O(1) lookup in the pre-computed matrix instead
      of calling encode() per star. Overhead: ~1ms vs ~665ms per star.

Outputs in --output directory:
  query_vectors.npy   → shape (N, 384) float32 — embedding matrix
  query_index.json    → {source_id: row_index}  — lookup index
  corpus_hash.txt     → sha256 of corpus — for cache invalidation

IMPORTANT: If stellar_corpus_v7.json or _build_query() change, the cache
is invalid. Regenerate with this script. engine.py checks the hash
automatically and falls back to real-time encode() on mismatch.

Usage:
    # Standard run from project root (login node, no GPU required)
    python3 pre_encode_queries.py

    # Explicit paths
    python3 pre_encode_queries.py \\
        --corpus  data/stellar_corpus_v7.json \\
        --output  data/rag_cache/ \\
        --model   all-MiniLM-L6-v2 \\
        --batch   256

    # Verify existing cache without regenerating
    python3 pre_encode_queries.py --verify-only

Version: 1.0 — STELLAR V7
  Adapted from MAGMA-01 pre_encode_queries.py v1.0
  Key changes:
    - star key is source_id (not galaxy_id)
    - _build_query() imported from STELLAR engine.py
    - project paths updated for STELLAR directory structure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Default paths ─────────────────────────────────────────────────────────────
# Adjust BASE_DIR if the script is moved inside a src/ subdirectory
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = BASE_DIR / "Data" / "stellar_corpus_v7.json"
DEFAULT_OUTPUT = BASE_DIR / "Data" / "rag_cache"
DEFAULT_MODEL  = "all-MiniLM-L6-v2"
DEFAULT_BATCH  = 256
DEFAULT_LOG    = BASE_DIR / "logs" / "pre_encode_queries.log"

# Cache file names — must match the constants in engine.py
FNAME_VECTORS = "query_vectors.npy"
FNAME_INDEX   = "query_index.json"
FNAME_HASH    = "corpus_hash.txt"


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pre_encode_queries")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Corpus hash ───────────────────────────────────────────────────────────────

def compute_corpus_hash(corpus_path: Path) -> str:
    """
    Compute SHA-256 of the corpus JSON in raw bytes.
    Used to detect corpus changes and invalidate the cache automatically.
    Must match the _hash_corpus() method in engine.py.
    """
    sha = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


# ── Cache verification ────────────────────────────────────────────────────────

def verify_cache(output_dir: Path, corpus_path: Path, log: logging.Logger) -> bool:
    """
    Verify that an existing cache is valid for the given corpus.

    Returns True if the cache exists and the corpus hash matches.
    """
    vectors_path = output_dir / FNAME_VECTORS
    index_path   = output_dir / FNAME_INDEX
    hash_path    = output_dir / FNAME_HASH

    if not all(p.exists() for p in [vectors_path, index_path, hash_path]):
        log.info("Cache incomplete or does not exist.")
        return False

    cached_hash  = hash_path.read_text(encoding="utf-8").strip()
    current_hash = compute_corpus_hash(corpus_path)

    if cached_hash != current_hash:
        log.warning(
            f"Corpus hash mismatch — cache is STALE.\n"
            f"  Cached  : {cached_hash[:32]}...\n"
            f"  Current : {current_hash[:32]}...\n"
            "  Regenerate with: python3 pre_encode_queries.py"
        )
        return False

    vectors = np.load(vectors_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    log.info(
        f"Cache VALID:\n"
        f"  Vectors : {vectors.shape}  dtype={vectors.dtype}\n"
        f"  Stars   : {len(index):,}\n"
        f"  Hash    : {cached_hash[:32]}...\n"
        f"  Path    : {output_dir}"
    )
    return True


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(corpus_path: Path, log: logging.Logger) -> list[dict]:
    """
    Load stellar_corpus_v7.json and return the list of HC contracts.

    Accepts two formats:
      - Direct list:  [{source_id: ..., hc_anchor: ..., ...}, ...]
      - Dict wrapper: {"stars": [...]} (in case format evolves)
    """
    log.info(f"Loading corpus: {corpus_path}")
    t0 = time.perf_counter()

    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        stars = data
    elif isinstance(data, dict):
        for key in ("stars", "contracts", "data", "corpus"):
            if key in data and isinstance(data[key], list):
                stars = data[key]
                log.info(f"Corpus loaded from key '{key}'.")
                break
        else:
            for v in data.values():
                if isinstance(v, list):
                    stars = v
                    break
            else:
                raise ValueError(
                    "No list of star contracts found in the corpus JSON. "
                    "Verify the format of stellar_corpus_v7.json."
                )
    else:
        raise ValueError(f"Unexpected corpus format: {type(data)}")

    elapsed = time.perf_counter() - t0
    log.info(f"Corpus loaded: {len(stars):,} stars in {elapsed:.2f}s.")
    return stars


# ── Query construction ────────────────────────────────────────────────────────

def build_all_queries(
    stars: list[dict],
    log: logging.Logger,
) -> tuple[list[str], list[str], list[int]]:
    """
    Build the hybrid query for each star using _build_query() from engine.py.

    Imports _build_query directly from the engine to guarantee that the
    pre-encoded vectors are IDENTICAL to what retrieve() would produce
    in real time. If the import fails, the script exits with an error.

    Returns
    -------
    queries     : list[str]  — query per star (same order as star_ids)
    star_ids    : list[str]  — source_id (as string) per star
    skipped_idx : list[int]  — indices of stars with empty queries
    """
    # Import _build_query from engine — consistency guarantee
    try:
        from engine import _build_query
        log.info("_build_query imported from engine.py (consistency guaranteed).")
    except ImportError as exc:
        raise ImportError(
            "Could not import _build_query from engine.py. "
            "Run from scripts/ or ensure engine.py is in the Python path."
        ) from exc

    queries:     list[str] = []
    star_ids:    list[str] = []
    skipped_idx: list[int] = []

    log.info(f"Building queries for {len(stars):,} stars...")
    t0 = time.perf_counter()

    for i, contract in enumerate(stars):
        # source_id is int64 in Gaia DR3 — store as string for JSON compatibility
        sid = str(contract.get("source_id", f"IDX_{i}"))
        try:
            q = _build_query(contract)
        except Exception as e:
            log.warning(f"[{sid}] Error in _build_query: {e} — using empty query.")
            q = ""

        if not q.strip():
            skipped_idx.append(i)
            queries.append("")   # placeholder to preserve index alignment
        else:
            queries.append(q)

        star_ids.append(sid)

    elapsed = time.perf_counter() - t0
    log.info(
        f"Queries built in {elapsed:.2f}s. "
        f"Empty: {len(skipped_idx)} / {len(stars):,}."
    )
    return queries, star_ids, skipped_idx


# ── Batch encoding ────────────────────────────────────────────────────────────

def encode_queries_batch(
    queries: list[str],
    model_name: str,
    batch_size: int,
    log: logging.Logger,
) -> np.ndarray:
    """
    Encode all queries in batch using SentenceTransformer.

    encode() parameters are IDENTICAL to those in engine.py retrieve():
      normalize_embeddings=True  → L2-normalized vectors (cosine = dot product)
      convert_to_numpy=True      → float32 ndarray

    Empty queries (skipped stars) are stored as zero vectors.
    engine.py detects zero vectors and returns empty context for those stars.

    Parameters
    ----------
    queries    : list of query strings. Empty strings are skipped.
    model_name : sentence-transformers model name.
    batch_size : batch size. 256 is optimal for CPU; reduce for low-RAM nodes.

    Returns
    -------
    np.ndarray of shape (N, embedding_dim) float32.
    """
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading embedding model: '{model_name}'...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    log.info(f"Model loaded in {time.perf_counter() - t0:.2f}s.")

    valid_mask    = [bool(q.strip()) for q in queries]
    valid_queries = [q for q, m in zip(queries, valid_mask) if m]
    n_total       = len(queries)
    n_valid       = len(valid_queries)
    n_skipped     = n_total - n_valid

    log.info(
        f"Batch encode: {n_valid:,} valid queries, "
        f"{n_skipped} empty (stored as zero vectors). "
        f"batch_size={batch_size}."
    )

    t0 = time.perf_counter()
    valid_embeddings = model.encode(
        valid_queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # CRITICAL: must match engine.py
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    embedding_dim = valid_embeddings.shape[1]
    log.info(
        f"Encoding complete: {n_valid:,} vectors, dim={embedding_dim} "
        f"in {elapsed:.1f}s "
        f"({elapsed / n_valid * 1000:.1f} ms/query average)."
    )

    # Reconstruct full matrix (N, dim) with zero vectors for empty queries
    all_embeddings = np.zeros((n_total, embedding_dim), dtype=np.float32)
    valid_iter = iter(valid_embeddings)
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            all_embeddings[i] = next(valid_iter)

    return all_embeddings


# ── Cache saving ──────────────────────────────────────────────────────────────

def save_cache(
    output_dir: Path,
    embeddings: np.ndarray,
    star_ids: list[str],
    corpus_hash: str,
    skipped_idx: list[int],
    log: logging.Logger,
) -> None:
    """
    Save the three cache files to output_dir.

    Files:
      query_vectors.npy  — (N, 384) float32 matrix
      query_index.json   — {source_id: row_index}
      corpus_hash.txt    — sha256 for cache invalidation in engine.py
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Embedding matrix
    vectors_path = output_dir / FNAME_VECTORS
    np.save(vectors_path, embeddings)
    size_mb = vectors_path.stat().st_size / 1024 / 1024
    log.info(f"Saved: {vectors_path}  ({size_mb:.1f} MB)")

    # 2. source_id → row_index lookup
    index = {sid: i for i, sid in enumerate(star_ids)}
    index_path = output_dir / FNAME_INDEX
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {index_path}  ({len(index):,} entries)")

    # 3. Corpus hash
    hash_path = output_dir / FNAME_HASH
    hash_path.write_text(corpus_hash, encoding="utf-8")
    log.info(f"Saved: {hash_path}  (sha256: {corpus_hash[:32]}...)")

    if skipped_idx:
        log.warning(
            f"{len(skipped_idx)} stars had empty queries (missing HC data). "
            f"Their vectors are zero — engine.py will return empty context. "
            f"First 10 indices: {skipped_idx[:10]}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STELLAR V7 — Pre-encode RAG queries for all stars in the corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run from project root
  python3 pre_encode_queries.py

  # Explicit paths
  python3 pre_encode_queries.py \\
      --corpus data/stellar_corpus_v7.json \\
      --output data/rag_cache/

  # Verify existing cache only
  python3 pre_encode_queries.py --verify-only

  # Force regeneration even if cache is valid
  python3 pre_encode_queries.py --force
        """,
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help=f"Path to stellar_corpus_v7.json. Default: {DEFAULT_CORPUS}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for cache files. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"sentence-transformers model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH,
        help=f"Batch size for encode(). Default: {DEFAULT_BATCH}",
    )
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG,
        help=f"Log file path. Default: {DEFAULT_LOG}",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify if existing cache is valid; do not regenerate.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate cache even if it already exists and is valid.",
    )
    args = parser.parse_args()

    log = setup_logging(args.log)
    log.info("=== pre_encode_queries.py v1.0 — STELLAR V7 — start ===")
    log.info(f"  Corpus : {args.corpus}")
    log.info(f"  Output : {args.output}")
    log.info(f"  Model  : {args.model}")
    log.info(f"  Batch  : {args.batch}")

    if not args.corpus.exists():
        log.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    # --verify-only mode
    if args.verify_only:
        is_valid = verify_cache(args.output, args.corpus, log)
        sys.exit(0 if is_valid else 1)

    # Skip if cache is already valid (unless --force)
    if not args.force:
        if verify_cache(args.output, args.corpus, log):
            log.info(
                "Cache already exists and is valid. "
                "Use --force to regenerate anyway."
            )
            log.info("=== pre_encode_queries.py — end (cache current) ===")
            return

    # ── Pre-encoding pipeline ─────────────────────────────────────────────────
    t_total = time.perf_counter()

    log.info("Computing corpus hash...")
    corpus_hash = compute_corpus_hash(args.corpus)
    log.info(f"sha256: {corpus_hash}")

    stars = load_corpus(args.corpus, log)
    queries, star_ids, skipped_idx = build_all_queries(stars, log)
    embeddings = encode_queries_batch(queries, args.model, args.batch, log)
    save_cache(args.output, embeddings, star_ids, corpus_hash, skipped_idx, log)

    elapsed_total = time.perf_counter() - t_total
    n_stars = len(stars)
    n_valid = n_stars - len(skipped_idx)

    log.info(
        f"\n{'='*60}\n"
        f"  STELLAR RAG — PRE-ENCODING SUMMARY\n"
        f"{'='*60}\n"
        f"  Stars processed   : {n_stars:,}\n"
        f"  Valid vectors     : {n_valid:,}\n"
        f"  Empty queries     : {len(skipped_idx)}\n"
        f"  Matrix shape      : {embeddings.shape}\n"
        f"  Total time        : {elapsed_total:.1f}s\n"
        f"  Cache saved to    : {args.output}\n"
        f"{'='*60}\n"
        f"  Next step: launch inference with inference_manager_v7.py\n"
        f"  RAGEngine will load the cache automatically.\n"
        f"{'='*60}"
    )
    log.info("=== pre_encode_queries.py v1.0 — STELLAR V7 — end ===")


if __name__ == "__main__":
    main()
