"""
engine.py — STELLAR RAG Engine
================================
Retrieval-Augmented Generation engine for the STELLAR HC+SC classifier.

Loads the STELLAR knowledge base (6 Markdown documents) once at startup,
encodes all chunks into a 384-dim embedding index, and retrieves the TOP_K
most semantically relevant chunks for each star's HC contract.

Adapted from MAGMA-01 RAGEngine (v1.1). Key differences:
  - _build_query() maps HC contract fields (spectral letter, flags, logg, chemistry)
    instead of galaxy morphology/BPT fields.
  - Knowledge base documents are stellar-classification-specific.
  - star_id key is source_id (int64 as string), not galaxy_id.

Model: sentence-transformers/all-MiniLM-L6-v2 (80MB, CPU-only, 384-dim)
Pre-cache: if cache_dir is provided with files from pre_encode_queries.py,
           retrieve() uses an O(1) fast path (~1ms/star) instead of encode()
           (~665ms/star).

Usage:
    from rag.engine import RAGEngine

    # Standard (no cache)
    rag = RAGEngine(knowledge_base_dir="knowledge_base/")

    # With pre-encoded query cache
    rag = RAGEngine(
        knowledge_base_dir="knowledge_base/",
        cache_dir="rag_cache/",
        corpus_path="stellar_corpus_v7.json",
    )

    context_block = rag.retrieve(hc_contract)  # inject into user prompt

Dependencies:
    pip install sentence-transformers numpy

Version: 1.0 — STELLAR V7
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 80MB, CPU-only, 384-dim
TOP_K_CHUNKS    = 3                      # chunks to retrieve per star
MIN_CHUNK_WORDS = 40                     # discard sections shorter than this

# Knowledge base documents to index (order does not affect retrieval)
KB_DOCUMENTS = [
    "mk_classification_rules.md",
    "subtype_calibration_guide.md",
    "luminosity_class_guide.md",
    "quality_flags_interpretation.md",
    "population_classification_guide.md",
    # "stellar_description_format.md",  # excluded: format template, not KB knowledge
]

# Cache file names — must match pre_encode_queries.py
CACHE_FNAME_VECTORS = "query_vectors.npy"
CACHE_FNAME_INDEX   = "query_index.json"
CACHE_FNAME_HASH    = "corpus_hash.txt"


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_document(text: str, source: str) -> list[dict]:
    """
    Split a Markdown document into semantic chunks by ## section headers.

    Each chunk includes the section title in its text so that the embedding
    captures the topic. Chunks shorter than MIN_CHUNK_WORDS are discarded.
    If the entire document has no ## sections but meets the word threshold,
    it is treated as a single chunk.
    """
    chunks = []
    sections = re.split(r"\n(?=## )", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section.split()) < MIN_CHUNK_WORDS:
            continue

        first_line = section.split("\n")[0].strip()
        title = first_line.lstrip("#").strip()

        chunks.append({
            "source": source,
            "title":  title,
            "text":   section,
            "label":  f"{source} — {title}",
        })

    # Fallback: whole document as one chunk
    if not chunks and len(text.split()) >= MIN_CHUNK_WORDS:
        chunks.append({
            "source": source,
            "title":  source,
            "text":   text.strip(),
            "label":  source,
        })

    return chunks


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_query(contract: dict) -> str:
    """
    Build a hybrid semantic query from a STELLAR HC contract.

    The query is a compact token string that maps the star's physical
    state onto the vocabulary used in the knowledge base documents.
    sentence-transformers/all-MiniLM-L6-v2 maps this string to the same
    semantic space as the KB chunks, enabling cosine retrieval.

    Parameters
    ----------
    contract : dict
        A single HC contract from stellar_corpus_v7.json. Expected top-level
        fields: source_id, hc_anchor, physical_vector, logical_flags,
        binary_diagnostics, quality_score.

    Returns
    -------
    str
        Space-separated token query (~15–35 tokens).
    """
    parts = []

    # ── HC anchor — spectral letter and population (locked by HC) ─────────────
    anchor = contract.get("hc_anchor", {})
    letter = anchor.get("mk_letter", "")
    if letter:
        parts.append(f"spectral_type:{letter}")

    population = anchor.get("population_group", "")
    if population:
        # Normalize to tokens compatible with KB vocabulary
        pop_token = population.replace(" ", "_")   # e.g. "Thin_Disk"
        parts.append(f"population:{pop_token}")

    # ── Physical vector ────────────────────────────────────────────────────────
    phys = contract.get("physical_vector", {})

    teff = phys.get("teff_k")
    if teff is not None:
        parts.append(f"teff:{int(teff)}")

    logg = phys.get("logg")
    if logg is not None:
        # Encode logg regime for luminosity-class retrieval
        if logg < 2.5:
            parts.append("logg:supergiant")
        elif logg < 3.5:
            parts.append("logg:giant")
        elif logg < 4.0:
            parts.append("logg:subgiant")
        else:
            parts.append("logg:dwarf")
        parts.append(f"logg:{logg:.2f}")

    metallicity = phys.get("metallicity")
    fe_h        = phys.get("fe_h")
    chem        = fe_h if fe_h is not None and fe_h != 0.0 else metallicity
    if chem is not None:
        if chem < -1.0:
            parts.append("metallicity:halo_metalpoor")
        elif chem < -0.2:
            parts.append("metallicity:thick_disk_subsolar")
        else:
            parts.append("metallicity:thin_disk_solar")
        parts.append(f"fe_h:{chem:.2f}")

    alpha_fe = phys.get("alpha_fe")
    if alpha_fe is not None:
        if alpha_fe >= 0.2:
            parts.append("alpha_enhanced")
        parts.append(f"alpha_fe:{alpha_fe:.2f}")

    v_tan = phys.get("v_tan")
    if v_tan is not None and v_tan > 200:
        parts.append("high_velocity_star")
        parts.append(f"v_tan:{v_tan:.0f}")

    # ── Logical flags ──────────────────────────────────────────────────────────
    flags = contract.get("logical_flags", {})

    if flags.get("is_binary_candidate"):
        parts.append("is_binary_candidate")
    if flags.get("is_giant"):
        parts.append("is_giant")
    if flags.get("is_high_velocity"):
        parts.append("is_high_velocity")
    if flags.get("has_emission"):
        parts.append("has_emission")
    if flags.get("fit_diverged"):
        parts.append("fit_diverged continuum_quality_flag")
    if not flags.get("continuum_is_stable", True):
        parts.append("continuum_unstable")
    if not flags.get("is_reliable_parallax", True):
        parts.append("parallax_unreliable")

    # ── A/B boundary — trigger logg-correction chunk ───────────────────────────
    if teff is not None and logg is not None and 9700 <= teff <= 10100:
        parts.append("AB_boundary logg_correction")

    # ── F subtype collapse warning — trigger subtype calibration chunk ─────────
    if letter == "F":
        parts.append("F_subtype_calibration subtype_collapse")

    return " ".join(parts)


# ── RAGEngine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    STELLAR RAG Engine.

    Loads and indexes the STELLAR knowledge base once at initialization.
    retrieve() returns a formatted context block ready to inject into the
    SC user prompt.

    Parameters
    ----------
    knowledge_base_dir : str | Path
        Directory containing the 6 KB Markdown files.
        Default: "knowledge_base/" relative to working directory.
    top_k : int
        Number of chunks to retrieve per star. Default: TOP_K_CHUNKS (3).
    model_name : str
        sentence-transformers model name. Default: all-MiniLM-L6-v2.
    cache_dir : str | Path | None
        Directory with pre-encoded query cache from pre_encode_queries.py.
        When valid, retrieve() uses an O(1) fast path. Default: None.
    corpus_path : str | Path | None
        Path to stellar_corpus_v7.json — used for cache hash validation.
        Default: None (skips hash check).
    """

    def __init__(
        self,
        knowledge_base_dir: str | Path = "knowledge_base/",
        top_k: int = TOP_K_CHUNKS,
        model_name: str = EMBEDDING_MODEL,
        cache_dir: Optional[str | Path] = None,
        corpus_path: Optional[str | Path] = None,
    ):
        self.kb_dir   = Path(knowledge_base_dir)
        self.top_k    = top_k
        self._chunks: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._query_cache = None   # (vectors_matrix, index_dict) | None

        log.info(f"RAGEngine: loading embedding model '{model_name}'...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        log.info("RAGEngine: embedding model loaded.")

        self._build_index()

        if cache_dir is not None:
            cp = Path(corpus_path) if corpus_path is not None else None
            self._query_cache = self._load_query_cache(Path(cache_dir), cp)

    # ── Index construction ─────────────────────────────────────────────────────

    def _build_index(self):
        """Load all KB documents, chunk them, and compute embeddings once."""
        all_chunks: list[dict] = []

        for doc_name in KB_DOCUMENTS:
            doc_path = self.kb_dir / doc_name
            if not doc_path.exists():
                log.warning(f"RAGEngine: document not found, skipping: {doc_path}")
                continue

            text   = doc_path.read_text(encoding="utf-8")
            chunks = _chunk_document(text, source=doc_name)
            all_chunks.extend(chunks)
            log.info(f"RAGEngine: {doc_name} → {len(chunks)} chunks")

        if not all_chunks:
            log.error(
                "RAGEngine: no chunks indexed. "
                "Verify --knowledge_base_dir and that the .md files contain ## sections."
            )
            self._chunks     = []
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            return

        self._chunks = all_chunks
        texts = [c["text"] for c in all_chunks]

        log.info(f"RAGEngine: encoding {len(texts)} chunks...")
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self._embeddings = embeddings.astype(np.float32)
        log.info(
            f"RAGEngine: index built — "
            f"{len(self._chunks)} chunks, dim={self._embeddings.shape[1]}."
        )

    # ── Query cache ────────────────────────────────────────────────────────────

    def _load_query_cache(
        self,
        cache_dir: Path,
        corpus_path: Optional[Path],
    ) -> Optional[tuple]:
        """
        Load pre-encoded query vectors from pre_encode_queries.py output.
        Returns (vectors_matrix, index_dict) if valid, None otherwise.
        Falls back gracefully — retrieve() will encode in real time.
        """
        vectors_path = cache_dir / CACHE_FNAME_VECTORS
        index_path   = cache_dir / CACHE_FNAME_INDEX
        hash_path    = cache_dir / CACHE_FNAME_HASH

        missing = [p for p in [vectors_path, index_path, hash_path] if not p.exists()]
        if missing:
            log.warning(
                f"RAGEngine: incomplete cache in '{cache_dir}'. "
                f"Missing: {[p.name for p in missing]}. "
                "Falling back to real-time encode(). "
                "Regenerate with: python3 pre_encode_queries.py"
            )
            return None

        if corpus_path is not None and corpus_path.exists():
            cached_hash  = hash_path.read_text(encoding="utf-8").strip()
            current_hash = self._hash_corpus(corpus_path)
            if cached_hash != current_hash:
                log.warning(
                    "RAGEngine: corpus hash mismatch — cache is STALE. "
                    "Falling back to real-time encode(). "
                    "Regenerate with: python3 pre_encode_queries.py"
                )
                return None

        try:
            vectors = np.load(vectors_path).astype(np.float32)
            with open(index_path, encoding="utf-8") as f:
                index: dict = json.load(f)
        except Exception as e:
            log.warning(f"RAGEngine: error loading cache: {e}. Falling back to encode().")
            return None

        expected_dim = self._embeddings.shape[1] if self._embeddings is not None else 384
        if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
            log.warning(
                f"RAGEngine: cache dimension {vectors.shape} does not match "
                f"model dimension {expected_dim}. Falling back to encode()."
            )
            return None

        log.info(
            f"RAGEngine: query cache loaded — "
            f"shape={vectors.shape}, stars={len(index):,}. Fast path active (~1ms/star)."
        )
        return (vectors, index)

    @staticmethod
    def _hash_corpus(corpus_path: Path) -> str:
        """SHA-256 of corpus file bytes. Must match pre_encode_queries.py."""
        sha = hashlib.sha256()
        with open(corpus_path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, contract: dict) -> tuple[str, str]:
        """
        Retrieve the TOP_K most relevant KB chunks for a star's HC contract.

        Uses the pre-encoded query cache (fast path, ~1ms) if available
        and the star's source_id is found in the cache index. Otherwise
        encodes the query in real time (~665ms).

        Parameters
        ----------
        contract : dict
            A single HC contract from stellar_corpus_v7.json.

        Returns
        -------
        context_block : str
            Formatted context block ready to inject into the SC user prompt.
            Empty string if the index is empty or the query is blank.
        top_label : str
            The ``label`` field of the highest-scoring KB chunk
            (e.g. ``"subtype_calibration_guide.md — Class F"``).
            Empty string when context_block is empty.
            Used by inference_manager_v7.py to stamp ``rag_top_chunk``
            in each SC result for Level 5 validation.
        """
        _EMPTY = ("", "")

        if self._embeddings is None or len(self._chunks) == 0:
            log.warning("RAGEngine: empty index, returning no context.")
            return _EMPTY

        star_id   = str(contract.get("source_id", "UNKNOWN"))
        query_vec: Optional[np.ndarray] = None

        # Fast path — O(1) lookup in pre-computed matrix
        if self._query_cache is not None:
            vectors_matrix, index_dict = self._query_cache
            if star_id in index_dict:
                row = index_dict[star_id]
                query_vec = vectors_matrix[row]   # shape (384,), already normalized
                if np.linalg.norm(query_vec) < 1e-6:
                    log.debug(f"RAGEngine [{star_id}]: zero vector in cache (empty query).")
                    return _EMPTY
                log.debug(f"RAGEngine [{star_id}]: fast path (cache hit).")
            else:
                log.debug(f"RAGEngine [{star_id}]: not in cache, falling back to encode().")

        # Fallback — real-time encode
        if query_vec is None:
            query = _build_query(contract)
            if not query.strip():
                log.debug(f"RAGEngine [{star_id}]: empty query, no context returned.")
                return _EMPTY
            query_vec = self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            log.debug(f"RAGEngine [{star_id}]: fallback encode() complete.")

        # Cosine similarity = dot product (embeddings are already L2-normalized)
        scores  = self._embeddings @ query_vec          # shape (N_chunks,)
        top_idx = np.argsort(scores)[::-1][: self.top_k]

        # Format context block for injection into user prompt
        context_parts = []
        for rank, idx in enumerate(top_idx, start=1):
            chunk = self._chunks[idx]
            score = float(scores[idx])
            context_parts.append(
                f"[Chunk {rank} | {chunk['label']} | score={score:.3f}]\n"
                f"{chunk['text']}"
            )

        context_block = "\n\n".join(context_parts)
        top_label     = self._chunks[top_idx[0]]["label"]

        log.debug(
            f"RAGEngine [{star_id}]: retrieved {self.top_k} chunks — "
            f"top score={float(scores[top_idx[0]]):.4f}, "
            f"label='{top_label}'"
        )
        return context_block, top_label

    # ── Introspection ──────────────────────────────────────────────────────────

    def index_info(self) -> dict:
        """Return a summary dict of the current index state."""
        return {
            "n_chunks":    len(self._chunks),
            "n_documents": len({c["source"] for c in self._chunks}),
            "embed_dim":   int(self._embeddings.shape[1]) if self._embeddings is not None else 0,
            "cache_active": self._query_cache is not None,
            "top_k":       self.top_k,
            "model":       EMBEDDING_MODEL,
        }