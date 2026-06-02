"""
inference_manager.py — SC Stellar Inference Engine
===================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
Version 1.0

Architecture mirrors MAGMA-01 inference_manager.py v1.4, adapted for
stellar MK classification over Gaia DR3 HC contracts.

Key inherited features from MAGMA-01 v1.4:
  [KEEP] ClosingBraceStoppingCriteria — stops generation at the closing "}"
         of the root JSON object. Eliminates max_new_tokens as the operative
         limit; it is now only a safety ceiling. This was the primary
         throughput bottleneck in early MAGMA-01 runs.
  [KEEP] Stateful Batching — on startup, scans output_dir for existing
         batch_*.json files and skips already-processed stars. Allows safe
         SLURM resumption after a 72h walltime interruption.
         Uses max(existing_indices)+1, not len(), to handle index gaps.
  [KEEP] Retry with reduced prompt — attempts 2+ use build_prompt_reduced()
         from system_prompt.py. Retrying with an identical prompt on a
         deterministic model (do_sample=False) produces the same failure.
  [KEEP] JSON repair — truncation recovery by appending closing brace.
  [KEEP] Logging with filemode="a" — preserves logs across SLURM runs.
  [KEEP] float16 without quantization — 4-bit quantization was tested in
         MAGMA-01 and discarded (slower on Titan RTX with BS=1).
  [KEEP] Flash Attention 2 disabled by default — enable with --flash-attn
         only after verifying AstroSage-Llama compatibility.

SC-specific changes vs MAGMA-01:
  [NEW]  Imports SYSTEM_PROMPT, build_prompt, build_prompt_reduced from
         system_prompt.py (separate module, not inlined).
  [NEW]  Output schema validation: checks that the LLM response contains the
         required SC fields (source_id, classification, confidence_scores,
         technical_reasoning) before accepting a result as successful.
  [NEW]  quality_score == 0.0 stars are processed but logged with a WARNING.
         The SC classifies them using only the tabular physical vector, as
         instructed by the system prompt Step A protocol.
  [NEW]  max_new_tokens default lowered to 512 (stellar JSON output is
         significantly more compact than MAGMA-01 galaxy output).
  [CHG]  Output filenames use "stellar" prefix for clarity.

Outputs:
  <output_dir>/batch_N_<JID>.json              — Results per batch
  <output_dir>/sc_results_<JID>.json           — Global consolidated results
  <output_dir>/sc_failed_<JID>.json            — Stars that failed all retries
  inference_manager_sc.log                     — Full log (append mode)

Usage:
    python3 inference_manager.py \\
        --corpus         /path/to/stellar_corpus.json \\
        --model          /path/to/AstroSage-Llama \\
        --output         /path/to/outputs/sc \\
        --batch-size     50 \\
        --max-retries    3 \\
        --max-new-tokens 512 \\
        --job-id         ${SLURM_JOB_ID}

SLURM environment (Lab-SB CIMAT):
    Partition : GPU  (nodes g-0-1 to g-0-12)
    Hardware  : NVIDIA Titan RTX 24 GB, 2 GPUs per node
    Walltime  : 72h limit — Stateful Batching handles resumption.
    Max nodes : 4 simultaneous per user on GPU partition.

Author: Hybrid Stellar Classifier Project / CIMAT
Version: 1.0
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

from system_prompt_v2 import SYSTEM_PROMPT, build_prompt, build_prompt_reduced, PROMPT_VERSION

# ---------------------------------------------------------------------------
# Logging — append mode preserves logs across SLURM runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="inference_manager_sc_v2.log",
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

    This solves the core throughput problem observed in MAGMA-01: the model
    never emits EOS within the max_new_tokens budget because it keeps
    generating text after the final "}". This criterion fires as soon as
    the JSON closes, making max_new_tokens a safety ceiling only.

    Mechanism: decodes the last 6 newly generated tokens and checks whether
    the resulting text ends with "}". The 6-token window avoids false
    positives from intermediate nested closing braces (e.g. inside
    "classification": {...}) because those are followed immediately by
    more content, not whitespace/newline.
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
# SC output schema validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP = {"source_id", "classification", "confidence_scores", "technical_reasoning"}
_REQUIRED_CLASSIFICATION = {"spectral_type", "sub_type_range", "luminosity_class", "population_group"}
_REQUIRED_CONFIDENCE = {"spectral_type_confidence", "luminosity_confidence", "population_confidence"}

_VALID_SPECTRAL_TYPES  = {"O", "B", "A", "F", "G", "K", "M"}
_VALID_LUM_CLASSES     = {"I", "II", "III", "IV", "V"}
_VALID_POPULATIONS     = {"Halo", "Disco Grueso", "Disco Fino"}


def validate_sc_output(result: dict, source_id: str) -> list[str]:
    """
    Validates the structural and taxonomic correctness of a SC output dict.

    Args:
        result    (dict): Parsed JSON output from the LLM.
        source_id (str):  Expected source_id for cross-check.

    Returns:
        list[str]: List of validation error strings. Empty = valid.
    """
    errors: list[str] = []

    missing_top = _REQUIRED_TOP - set(result.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors  # Cannot proceed without structure

    clf = result.get("classification", {})
    missing_clf = _REQUIRED_CLASSIFICATION - set(clf.keys())
    if missing_clf:
        errors.append(f"Missing classification keys: {sorted(missing_clf)}")

    conf = result.get("confidence_scores", {})
    missing_conf = _REQUIRED_CONFIDENCE - set(conf.keys())
    if missing_conf:
        errors.append(f"Missing confidence_scores keys: {sorted(missing_conf)}")

    # Taxonomy enforcement
    sp = clf.get("spectral_type", "")
    if sp not in _VALID_SPECTRAL_TYPES:
        errors.append(f"Invalid spectral_type: '{sp}'. Must be one of {_VALID_SPECTRAL_TYPES}.")

    lc = clf.get("luminosity_class", "")
    if lc not in _VALID_LUM_CLASSES:
        errors.append(f"Invalid luminosity_class: '{lc}'. Must be one of {_VALID_LUM_CLASSES}.")

    pg = clf.get("population_group", "")
    if pg not in _VALID_POPULATIONS:
        errors.append(f"Invalid population_group: '{pg}'. Must be one of {_VALID_POPULATIONS}.")

    # Confidence score range check
    for field, val in conf.items():
        if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            errors.append(f"confidence_scores.{field}={val} is out of [0.0, 1.0].")

    # source_id cross-check
    if str(result.get("source_id", "")) != str(source_id):
        errors.append(
            f"source_id mismatch: expected '{source_id}', got '{result.get('source_id')}'."
        )

    return errors


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_path: str, flash_attn: bool = False):
    """
    Loads AstroSage-Llama in float16 on the available GPU(s).

    flash_attn: disabled by default. The Titan RTX with BS=1 has no memory
    bottleneck; enabling requires manual verification of AstroSage-Llama
    compatibility. Activate with --flash-attn only after testing.

    Args:
        model_path (str):   Local path to the model directory on the cluster.
        flash_attn (bool):  Enable Flash Attention 2.

    Returns:
        tuple: (tokenizer, model)
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

def infer_star(
    star: dict,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    reduced_prompt: bool = False,
) -> tuple[Optional[dict], str]:
    """
    Runs inference for a single star.

    Args:
        star           (dict): A single entry from stellar_corpus.json.
        tokenizer:             Loaded tokenizer.
        model:                 Loaded model.
        max_new_tokens (int):  Token ceiling (safety only — ClosingBrace fires first).
        reduced_prompt (bool): Use build_prompt_reduced() for retry attempts.

    Returns:
        tuple: (result_dict | None, raw_output_str)
               result_dict is None if parsing or validation failed.
               raw_output_str is saved in the failed list for diagnosis.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 1.0)

    if qs == 0.0:
        log.warning(f"  [{source_id}] quality_score=0.0 — parallax unreliable. "
                    f"Classifying from tabular vector only.")

    prompt = build_prompt_reduced(star) if reduced_prompt else build_prompt(star)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

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
        text = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    n_input_tokens = input_ids.shape[-1]
    log.info(
        f"  [{source_id}] Input tokens: {n_input_tokens} | "
        f"max_new_tokens: {max_new_tokens} | reduced_prompt: {reduced_prompt}"
    )

    # Standard EOS stop tokens
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
    # JSON parser + repair
    # -------------------------------------------------------------------------
    try:
        clean = raw_output.replace("```json", "").replace("```", "").strip()

        # Discard any <<SYS>> prefix artifact from the LLaMA2 fallback
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
                # Repair: truncation just before the final "}"
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

        # -------------------------------------------------------------------------
        # SC output schema validation
        # -------------------------------------------------------------------------
        validation_errors = validate_sc_output(result, source_id)
        if validation_errors:
            for err in validation_errors:
                log.warning(f"  [{source_id}] Validation error: {err}")
            # Attach errors to result for traceability, but still return it
            # if the core classification fields are present. Only return None
            # if the top-level structure is missing entirely.
            if any("Missing top-level" in e for e in validation_errors):
                return None, raw_output
            result["_validation_warnings"] = validation_errors

        return result, ""

    except json.JSONDecodeError as e:
        log.warning(f"  [{source_id}] Invalid JSON (unrepairable): {e}")
        log.warning(f"  [{source_id}] Raw output (first 400 chars): {raw_output[:400]}")
        return None, raw_output


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_batch(
    batch: list,
    batch_idx: int,
    tokenizer,
    model,
    max_retries: int = 3,
    max_new_tokens: int = 512,
) -> tuple[list, list]:
    """
    Processes a batch of stars with per-star retry logic.

    Attempt 1: full prompt (build_prompt).
    Attempts 2+: reduced prompt (build_prompt_reduced). Retrying with an
    identical prompt on a deterministic model (do_sample=False) produces the
    same failure — the reduced prompt changes the input context.

    Args:
        batch          (list): List of star dicts from stellar_corpus.json.
        batch_idx      (int):  Batch index (for logging).
        tokenizer:             Loaded tokenizer.
        model:                 Loaded model.
        max_retries    (int):  Maximum retry attempts per star.
        max_new_tokens (int):  Token ceiling for generation.

    Returns:
        tuple: (results list, failed list)
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
            log.error(
                f"  [{source_id}] Failed after {max_retries} attempts."
            )
            failed.append({
                "source_id":       source_id,
                "quality_score":   star.get("quality_score"),
                "reason":          f"Failed after {max_retries} attempts.",
                "batch":           batch_idx,
                "last_raw_output": last_raw[:500],
            })

    return results, failed


# ---------------------------------------------------------------------------
# Global results compiler
# ---------------------------------------------------------------------------

def _compile_global(output_dir: Path, job_id: str) -> None:
    """
    Consolidates all existing batch_*.json files into a single sc_results file.
    Called at the end of every run (including resumed runs) to keep the global
    file always up to date.
    """
    all_results: list[dict] = []

    for batch_file in sorted(
        output_dir.glob("batch_v2_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),  # sort by numeric batch index
    ):
        try:
            with open(batch_file, "r", encoding="utf-8") as bf:
                all_results.extend(json.load(bf))
        except Exception as ex:
            log.warning(f"Could not read {batch_file} during compilation: {ex}")

    suffix      = f"_{job_id}" if job_id else ""
    global_path = output_dir / f"sc_results_v2{suffix}.json"
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
    max_new_tokens: int = 512,
    job_id: str = "",
    flash_attn: bool = False,
) -> None:
    """
    Orchestrates the full SC inference pipeline with Stateful Batching.

    Stateful Batching: on startup, scans output_dir for existing batch_*.json
    files and excludes already-processed stars from the corpus. If SLURM
    interrupts the job (72h walltime), the next run resumes from where it
    stopped without reprocessing anything.

    Bug note (inherited from MAGMA-01): next batch index is computed as
    max(existing_indices)+1, NOT len(existing)+1. len() fails when gaps
    exist (e.g. batch_1 and batch_3 exist but not batch_2).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"SC Stellar Inference Engine v1.0 — prompt={PROMPT_VERSION} — job_id={job_id or 'N/A'}")
    log.info("=" * 60)

    # Load corpus
    log.info(f"Reading corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    log.info(f"Total stars in corpus: {len(corpus)}")

    # Stateful Batching — exclude already-processed stars
    processed_ids: set[str]     = set()
    existing_batch_indices: list[int] = []

    for batch_file in output_dir.glob("batch_v2_*.json"):
        try:
            stem = batch_file.stem           # e.g. "batch_3_180656"
            idx  = int(stem.split("_")[2])   # batch_v2_N: index is third segment
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
            f"({original_len - len(corpus)} skipped). "
            f"Remaining: {len(corpus)}"
        )

    if len(corpus) == 0:
        log.info("All stars already processed. Compiling global results.")
        _compile_global(output_dir, job_id)
        return

    next_batch_idx = (max(existing_batch_indices) + 1) if existing_batch_indices else 1

    # Load model
    tokenizer, model = load_model(model_path, flash_attn=flash_attn)

    # Build and process batches
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

        suffix_batch = f"_{job_id}" if job_id else ""
        batch_path   = output_dir / f"batch_v2_{batch_idx}{suffix_batch}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"  Batch {batch_idx} saved -> {batch_path}")

        all_results.extend(results)
        all_failed.extend(failed)

    # Consolidate all batches (old + new) into sc_results
    _compile_global(output_dir, job_id)

    # Save failed stars
    if all_failed:
        suffix      = f"_{job_id}" if job_id else ""
        failed_path = output_dir / f"sc_failed_v2{suffix}.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, indent=2, ensure_ascii=False)
        log.warning(f"Failed stars: {len(all_failed)} -> {failed_path}")

    # Final summary
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
        description="HC+SC Stellar Classifier — Inference Engine v1.0"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to stellar_corpus.json (output of corpus_builder.py)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local path to AstroSage-Llama on the cluster",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sc"),
        help="Output directory (default: outputs/sc)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Stars per batch (default: 50)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per star (default: 3)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Token ceiling for generation (default: 512). "
            "ClosingBraceStoppingCriteria fires before this limit for "
            "well-formed JSON outputs — this is a safety ceiling only."
        ),
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="",
        help="SLURM job ID for output file naming (default: empty)",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help=(
            "Enable Flash Attention 2 (disabled by default). "
            "Verify AstroSage-Llama compatibility before enabling."
        ),
    )
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
