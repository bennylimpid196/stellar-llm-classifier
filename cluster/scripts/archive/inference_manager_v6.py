"""
inference_manager_v6.py — SC Stellar Inference Engine V6
=========================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
System: STELLAR

Changes vs V5
-------------
V6-INF-1 | Import from system_prompt_v6
    Uses build_messages(star) instead of separate SYSTEM_PROMPT + build_prompt().
    build_messages() encapsulates the soft A/B anchor logic (V6-CHG-1) and the
    notable_features rules (V6-CHG-3) inside the prompt builder.

V6-INF-2 | max_new_tokens default raised to 1024
    V5 post-mortem: 181/498 failures (36.3%) were caused by JSON truncation.
    The stellar_description field (physical_profile + population_context +
    notable_features) consistently exceeded the 512-token ceiling introduced
    in V5. Raising to 1024 gives comfortable headroom for well-formed output
    while ClosingBraceStoppingCriteria still fires early for clean outputs.

V6-INF-3 | All output filenames use v6 suffix
    batch_v6_N_JID.json, sc_results_v6_JID.json, sc_failed_v6_JID.json
    Log file: inference_manager_sc_v6.log
    Stateful batching glob pattern updated to batch_v6_*.json

V6-INF-4 | prompt_version field in failed entries
    Failed star dicts now include prompt_version: "v6" for traceability.

Inherited from V5 (unchanged)
------------------------------
  ClosingBraceStoppingCriteria  — stops at closing "}" of root JSON
  Stateful Batching             — skips already-processed stars on resume
  Retry with reduced prompt     — attempt 2+ use build_messages_reduced()
  JSON repair                   — truncation recovery via closing-brace append
  Logging filemode="a"          — preserves logs across SLURM runs
  float16 without quantization  — faster on Titan RTX with BS=1
  Flash Attention 2 opt-in      — --flash-attn flag

Outputs
-------
  <output_dir>/batch_v6_N_<JID>.json     — Results per batch
  <output_dir>/sc_results_v6_<JID>.json  — Consolidated global results
  <output_dir>/sc_failed_v6_<JID>.json   — Stars that failed all retries
  inference_manager_sc_v6.log            — Full log (append mode)

Usage
-----
    python3 inference_manager_v6.py \\
        --corpus         /path/to/stellar_corpus_v5.json \\
        --model          /path/to/AstroSage-8B \\
        --output         /path/to/outputs/sc \\
        --batch-size     50 \\
        --max-retries    3 \\
        --max-new-tokens 1024 \\
        --job-id         ${SLURM_JOB_ID}

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 2.0
"""

import json
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

# V6-INF-1: import from system_prompt_v6
from system_prompt_v6 import build_messages, build_system_prompt, build_user_prompt

PROMPT_VERSION = "v6"

# ---------------------------------------------------------------------------
# Logging — append mode preserves logs across SLURM runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="inference_manager_sc_v6.log",   # V6-INF-3
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_console)


# ---------------------------------------------------------------------------
# StoppingCriteria — stops generation at the closing "}" of the root JSON
# ---------------------------------------------------------------------------

class ClosingBraceStoppingCriteria(StoppingCriteria):
    """
    Stops generation when the newly generated tokens end with "}",
    signalling the closure of the root JSON object.

    Inherited from V5/MAGMA-01 v1.4 — unchanged.
    """

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer  = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        new_ids = input_ids[0][self.prompt_len:]
        if len(new_ids) < 2:
            return False
        tail_ids  = new_ids[-6:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
        return tail_text.rstrip().endswith("}")


# ---------------------------------------------------------------------------
# SC output schema validation — unchanged from V5
# ---------------------------------------------------------------------------

_REQUIRED_TOP          = {"source_id", "classification", "confidence_scores",
                           "technical_reasoning", "stellar_description"}
_REQUIRED_CLASSIFICATION = {"spectral_type", "sub_type_range",
                             "luminosity_class", "population_group"}
_REQUIRED_CONFIDENCE   = {"spectral_type_confidence", "luminosity_confidence",
                           "population_confidence"}
_REQUIRED_DESCRIPTION  = {"physical_profile", "population_context", "notable_features"}

_VALID_SPECTRAL_TYPES  = {"O", "B", "A", "F", "G", "K", "M"}
_VALID_LUM_CLASSES     = {"I", "II", "III", "IV", "V"}
_VALID_POPULATIONS     = {"Halo", "Disco Grueso", "Disco Fino"}


def validate_sc_output(result: dict, source_id: str) -> list[str]:
    """
    Validates structural and taxonomic correctness of a SC output dict.
    Returns list of error strings; empty = valid.
    """
    errors: list[str] = []

    missing_top = _REQUIRED_TOP - set(result.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors

    clf = result.get("classification", {})
    missing_clf = _REQUIRED_CLASSIFICATION - set(clf.keys())
    if missing_clf:
        errors.append(f"Missing classification keys: {sorted(missing_clf)}")

    conf = result.get("confidence_scores", {})
    missing_conf = _REQUIRED_CONFIDENCE - set(conf.keys())
    if missing_conf:
        errors.append(f"Missing confidence_scores keys: {sorted(missing_conf)}")

    sp = clf.get("spectral_type", "")
    if sp not in _VALID_SPECTRAL_TYPES:
        errors.append(f"Invalid spectral_type: '{sp}'. Must be one of {_VALID_SPECTRAL_TYPES}.")

    lc = clf.get("luminosity_class", "")
    if lc not in _VALID_LUM_CLASSES:
        errors.append(f"Invalid luminosity_class: '{lc}'. Must be one of {_VALID_LUM_CLASSES}.")

    pg = clf.get("population_group", "")
    if pg not in _VALID_POPULATIONS:
        errors.append(f"Invalid population_group: '{pg}'. Must be one of {_VALID_POPULATIONS}.")

    for field, val in conf.items():
        if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            errors.append(f"confidence_scores.{field}={val} out of [0.0, 1.0].")

    sd = result.get("stellar_description", {})
    missing_sd = _REQUIRED_DESCRIPTION - set(sd.keys())
    if missing_sd:
        errors.append(f"Missing stellar_description keys: {sorted(missing_sd)}")
    else:
        for field in _REQUIRED_DESCRIPTION:
            if not sd.get(field):
                errors.append(f"stellar_description.{field} is empty.")

    if str(result.get("source_id", "")) != str(source_id):
        errors.append(
            f"source_id mismatch: expected '{source_id}', got '{result.get('source_id')}'."
        )

    return errors


# ---------------------------------------------------------------------------
# Model loader — unchanged from V5
# ---------------------------------------------------------------------------

def load_model(model_path: str, flash_attn: bool = False):
    """
    Loads AstroSage-8B in float16 on the available GPU(s).

    flash_attn: disabled by default. Enable with --flash-attn only after
    verifying AstroSage-Llama compatibility.
    """
    log.info(f"Loading model from: {model_path} | flash_attn={flash_attn}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    kwargs = dict(
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    log.info("Model loaded successfully.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Single-star inference
# ---------------------------------------------------------------------------

def _build_messages_reduced(star: dict) -> list[dict]:
    """
    Reduced prompt for retry attempts (attempts 2+).

    Uses the same system prompt but strips spectral_diagnostics (CaT EWs,
    BP/RP SNR) from the user prompt to reduce token count and break out of
    deterministic failure modes. The soft A/B anchor logic (V6-CHG-1) and
    notable_features rules (V6-CHG-3) are preserved in both full and reduced.
    """
    import copy
    star_reduced = copy.deepcopy(star)
    # Remove high-token spectral diagnostics for the reduced attempt
    star_reduced["spectral_summary"] = {}
    system_msg = build_system_prompt()
    user_msg   = build_user_prompt(star_reduced)
    # Append reduction notice
    user_msg += "\n\n[Reduced prompt — spectral diagnostics omitted for retry.]"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


def infer_star(
    star: dict,
    tokenizer,
    model,
    max_new_tokens: int = 1024,   # V6-INF-2: raised from 512
    reduced_prompt: bool = False,
) -> tuple[Optional[dict], str]:
    """
    Runs inference for a single star.

    Args:
        star           (dict): Entry from stellar_corpus_v5.json.
        tokenizer:             Loaded tokenizer.
        model:                 Loaded model.
        max_new_tokens (int):  Token ceiling (safety only — ClosingBrace fires first).
        reduced_prompt (bool): Use reduced prompt for retry attempts.

    Returns:
        tuple: (result_dict | None, raw_output_str)
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 1.0)

    if qs == 0.0:
        log.warning(f"  [{source_id}] quality_score=0.0 — parallax unreliable. "
                    f"Classifying from tabular vector only.")

    # V6-INF-1: use build_messages / _build_messages_reduced
    messages = _build_messages_reduced(star) if reduced_prompt else build_messages(star)

    # Tokenize — LLaMA2 manual fallback if apply_chat_template is unsupported
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        log.warning(
            f"  [{source_id}] apply_chat_template failed — using manual LLaMA2 fallback."
        )
        system_content = messages[0]["content"]
        user_content   = messages[1]["content"]
        text = (
            f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    n_input_tokens = input_ids.shape[-1]
    log.info(
        f"  [{source_id}] Input tokens: {n_input_tokens} | "
        f"max_new_tokens: {max_new_tokens} | reduced_prompt: {reduced_prompt}"
    )

    stop_token_ids = list({
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    } - {None, -1})

    stopping_criteria = StoppingCriteriaList([
        ClosingBraceStoppingCriteria(tokenizer, prompt_len=n_input_tokens)
    ])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0][n_input_tokens:]
    n_output   = len(new_tokens)
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    log.info(f"  [{source_id}] Output tokens generated: {n_output}")

    if n_output >= max_new_tokens:
        log.warning(
            f"  [{source_id}] Reached max_new_tokens ({max_new_tokens}). "
            f"ClosingBraceCriteria did not fire — JSON likely truncated."
        )

    # -------------------------------------------------------------------------
    # JSON parser + repair — unchanged from V5
    # -------------------------------------------------------------------------
    try:
        clean = raw_output.replace("```json", "").replace("```", "").strip()

        start = clean.find("{")
        end   = clean.rfind("}") + 1

        if start == -1 or end <= start:
            result = json.loads(clean)
        else:
            json_str   = clean[start:end]
            extra_text = clean[end:].strip()

            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                repaired = json_str.rstrip().rstrip(",")
                if not repaired.endswith("}"):
                    repaired = repaired + "\n}"
                result = json.loads(repaired)
                result["_repaired"] = True
                log.warning(f"  [{source_id}] JSON truncated — successfully repaired.")

            if extra_text:
                log.debug(
                    f"  [{source_id}] Discarding {len(extra_text)} chars after JSON close."
                )

        # Schema validation
        validation_errors = validate_sc_output(result, source_id)
        if validation_errors:
            for err in validation_errors:
                log.warning(f"  [{source_id}] Validation error: {err}")
            if any("Missing top-level" in e for e in validation_errors):
                return None, raw_output
            result["_validation_warnings"] = validation_errors

        # Stamp prompt version
        result["prompt_version"] = PROMPT_VERSION

        return result, ""

    except json.JSONDecodeError as e:
        log.warning(f"  [{source_id}] Invalid JSON (unrepairable): {e}")
        log.warning(f"  [{source_id}] Raw output (first 400 chars): {raw_output[:400]}")
        return None, raw_output


# ---------------------------------------------------------------------------
# Batch processor — unchanged from V5
# ---------------------------------------------------------------------------

def process_batch(
    batch: list,
    batch_idx: int,
    tokenizer,
    model,
    max_retries: int = 3,
    max_new_tokens: int = 1024,
) -> tuple[list, list]:
    """
    Processes a batch of stars with per-star retry logic.

    Attempt 1: full prompt (build_messages).
    Attempts 2+: reduced prompt (_build_messages_reduced).
    """
    results: list[dict] = []
    failed:  list[dict] = []

    for star in batch:
        source_id = str(star.get("source_id", "UNKNOWN"))
        success   = False
        last_raw  = ""

        for attempt in range(1, max_retries + 1):
            reduced = (attempt > 1)
            log.info(
                f"  [{source_id}] Attempt {attempt}/{max_retries} | "
                f"reduced_prompt={reduced}"
            )

            try:
                result, raw = infer_star(
                    star, tokenizer, model,
                    max_new_tokens=max_new_tokens,
                    reduced_prompt=reduced,
                )
                last_raw = raw

                if result is not None:
                    results.append(result)
                    clf = result.get("classification", {})
                    log.info(
                        f"  [{source_id}] SUCCESS — "
                        f"{clf.get('spectral_type','?')}"
                        f"{clf.get('sub_type_range','?')} "
                        f"{clf.get('luminosity_class','?')} | "
                        f"{clf.get('population_group','?')}"
                    )
                    success = True
                    break
                else:
                    log.warning(f"  [{source_id}] Null result on attempt {attempt}.")

            except Exception as e:
                last_raw = str(e)
                log.error(f"  [{source_id}] Exception on attempt {attempt}: {e}")

            if attempt < max_retries:
                time.sleep(2)

        if not success:
            log.error(f"  [{source_id}] Failed after {max_retries} attempts.")
            failed.append({
                "source_id":       source_id,
                "quality_score":   star.get("quality_score"),
                "prompt_version":  PROMPT_VERSION,    # V6-INF-4
                "reason":          f"Failed after {max_retries} attempts.",
                "batch":           batch_idx,
                "last_raw_output": last_raw[:500],
            })

    return results, failed


# ---------------------------------------------------------------------------
# Global results compiler — V6-INF-3: glob and filenames updated
# ---------------------------------------------------------------------------

def _compile_global(output_dir: Path, job_id: str) -> None:
    """
    Consolidates all batch_v6_*.json files into sc_results_v6_JID.json.
    """
    all_results: list[dict] = []

    for batch_file in sorted(
        output_dir.glob("batch_v6_*.json"),
        key=lambda p: int(p.stem.split("_")[2]),  # batch_v6_N_JID: index is [2]
    ):
        try:
            with open(batch_file, "r", encoding="utf-8") as bf:
                all_results.extend(json.load(bf))
        except Exception as ex:
            log.warning(f"Could not read {batch_file} during compilation: {ex}")

    suffix      = f"_{job_id}" if job_id else ""
    global_path = output_dir / f"sc_results_v6{suffix}.json"
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"Global results compiled: {len(all_results)} stars -> {global_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_inference(
    corpus_path: Path,
    model_path: str,
    output_dir: Path,
    batch_size: int = 50,
    max_retries: int = 3,
    max_new_tokens: int = 1024,   # V6-INF-2
    job_id: str = "",
    flash_attn: bool = False,
) -> None:
    """
    Orchestrates the full SC inference pipeline with Stateful Batching.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(
        f"SC Stellar Inference Engine v2.0 — prompt={PROMPT_VERSION} — job_id={job_id or 'N/A'}"
    )
    log.info("=" * 60)

    log.info(f"Reading corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    log.info(f"Total stars in corpus: {len(corpus)}")

    # Stateful Batching — V6-INF-3: glob updated to batch_v6_*
    processed_ids: set[str]        = set()
    existing_batch_indices: list[int] = []

    for batch_file in output_dir.glob("batch_v6_*.json"):
        try:
            stem = batch_file.stem           # e.g. "batch_v6_3_184200"
            idx  = int(stem.split("_")[2])   # index is third segment
            existing_batch_indices.append(idx)
            with open(batch_file, "r", encoding="utf-8") as bf:
                for item in json.load(bf):
                    processed_ids.add(str(item.get("source_id")))
        except Exception as ex:
            log.warning(f"Could not read existing batch file {batch_file}: {ex}")

    if processed_ids:
        original_len = len(corpus)
        corpus = [s for s in corpus if str(s.get("source_id")) not in processed_ids]
        log.info(
            f"Stateful Batching: {len(processed_ids)} stars already processed "
            f"({original_len - len(corpus)} skipped). Remaining: {len(corpus)}"
        )

    if len(corpus) == 0:
        log.info("All stars already processed. Compiling global results.")
        _compile_global(output_dir, job_id)
        return

    next_batch_idx = (max(existing_batch_indices) + 1) if existing_batch_indices else 1

    tokenizer, model = load_model(model_path, flash_attn=flash_attn)

    batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    log.info(
        f"Batches to process: {len(batches)} x ~{batch_size} stars "
        f"| max_new_tokens: {max_new_tokens}"
    )

    all_results: list[dict] = []
    all_failed:  list[dict] = []

    for i, batch in enumerate(batches):
        batch_idx = next_batch_idx + i
        log.info(f"=== Batch {batch_idx} ({len(batch)} stars) ===")

        results, failed = process_batch(
            batch, batch_idx, tokenizer, model, max_retries, max_new_tokens
        )

        # V6-INF-3: batch filename uses v6
        suffix_batch = f"_{job_id}" if job_id else ""
        batch_path   = output_dir / f"batch_v6_{batch_idx}{suffix_batch}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"  Batch {batch_idx} saved -> {batch_path}")

        all_results.extend(results)
        all_failed.extend(failed)

    _compile_global(output_dir, job_id)

    if all_failed:
        suffix      = f"_{job_id}" if job_id else ""
        failed_path = output_dir / f"sc_failed_v6{suffix}.json"  # V6-INF-3
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, indent=2, ensure_ascii=False)
        log.warning(f"Failed stars: {len(all_failed)} -> {failed_path}")

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"  Stars processed this run : {len(all_results)}")
    log.info(f"  Failed this run          : {len(all_failed)}")
    log.info(f"  Batches processed        : {len(batches)}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HC+SC Stellar Classifier — Inference Engine V6"
    )
    parser.add_argument("--corpus",         type=Path, required=True,
                        help="Path to stellar_corpus_v5.json")
    parser.add_argument("--model",          type=str,  required=True,
                        help="Local path to AstroSage-8B on the cluster")
    parser.add_argument("--output",         type=Path, default=Path("outputs/sc"),
                        help="Output directory (default: outputs/sc)")
    parser.add_argument("--batch-size",     type=int,  default=50,
                        help="Stars per batch (default: 50)")
    parser.add_argument("--max-retries",    type=int,  default=3,
                        help="Max retry attempts per star (default: 3)")
    parser.add_argument("--max-new-tokens", type=int,  default=1024,
                        help="Token ceiling for generation (default: 1024). "
                             "ClosingBraceStoppingCriteria fires before this "
                             "for well-formed outputs — safety ceiling only.")
    parser.add_argument("--job-id",         type=str,  default="",
                        help="SLURM job ID for output file naming")
    parser.add_argument("--flash-attn",     action="store_true", default=False,
                        help="Enable Flash Attention 2 (verify compatibility first)")
    args = parser.parse_args()

    if not args.corpus.exists():
        log.error(f"Corpus not found: {args.corpus}")
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")

    run_inference(
        corpus_path    = args.corpus,
        model_path     = args.model,
        output_dir     = args.output,
        batch_size     = args.batch_size,
        max_retries    = args.max_retries,
        max_new_tokens = args.max_new_tokens,
        job_id         = args.job_id,
        flash_attn     = args.flash_attn,
    )


if __name__ == "__main__":
    main()
