from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import Config
from .state import StateDB

# ─── Faulty document detection ────────────────────────────────────────────────

# CJK Unified Ideographs (basic block, covers virtually all common Chinese chars)
_CJK_RANGE = range(0x4E00, 0xA000)

def _contains_cjk(text: str) -> bool:
    return any(ord(ch) in _CJK_RANGE for ch in text)


# Known Whisper mishearings for this course.
# Key = bad term Whisper wrote, value = what was actually said (documentation only).
# Add new entries here as they are discovered.
_KNOWN_BAD_TERMS: dict[str, str] = {
    "DOCP":   "DHCP",
    "DOCPS":  "DHCP",
    "COMFIG": "CONFIG",
    "HTSP":   "HTTP",
    "HTTB":   "HTTP",
    "VELAN":  "VLAN",
    "UEFE":   "UEFI",
    "UEFY":   "UEFI",
    "CTFS":   "NTFS",
    "NTFLS":  "NTFS",
    "EPFIRE": "IPFIRE",
    "IPW6":   "IPv6",
    "IPvS6":  "IPv6",
    "IPw6":   "IPv6",
    "IPv4S":  "IPv4",
    "IPv4s":  "IPv4",
    "PIXE":   "PXE",
    "SSHI":   "SSH",
    "CSVD":   "CSV",
    "EFAT":   "EXFAT",
    "EXD2":   "EXT2",
    "EXD4":   "EXT4",
    "IEEI":   "IEEE",
    "LTLM":   "NTLM",
    "NAWT":   "NAT",
    "OSPFO":  "OSPF",
    "RDPL":   "RDP",
    "SCASIA": "SCSI",
    "SCASSID": "SCSI",
    "TSPIP":  "TCPIP",
    "VODX":   "VHDX",
    "VVPN":   "VPN",
    "WACC":   "WAC",
}

# Pre-compiled whole-word, case-insensitive patterns (built once at import time).
_BAD_TERM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in _KNOWN_BAD_TERMS
]


def get_bad_terms(text: str) -> list[str]:
    """Return a list of unique known-bad terms found in *text* using strict word boundaries."""
    found = []
    # We iterate over the dictionary keys to return the original term (e.g. "EFAT")
    # rather than whatever case was used in the text.
    for term, pattern in zip(_KNOWN_BAD_TERMS, _BAD_TERM_PATTERNS):
        if pattern.search(text):
            found.append(term)
    return found


def _is_faulty(text: str) -> bool:
    """Return True if *text* contains CJK characters or any known-bad Whisper term."""
    if _contains_cjk(text):
        return True
    return any(p.search(text) for p in _BAD_TERM_PATTERNS)


def _transcript_needs_correction(transcript: str) -> bool:
    """Return True if the raw Whisper transcript contains any known-bad term."""
    return any(p.search(transcript) for p in _BAD_TERM_PATTERNS)


def _summaries_are_clean(brief: str, detailed: str) -> bool:
    """Return True if neither summary contains a known-bad term.

    A clean summary can be used as a trusted reference to correct the transcript.
    If the summaries are also contaminated the LLM correction path is not safe to use.
    """
    combined = brief + " " + detailed
    return not any(p.search(combined) for p in _BAD_TERM_PATTERNS)


async def correct_transcript(
    transcript: str,
    brief: str,
    detailed: str,
    cfg: Config,
    logger: logging.Logger,
) -> str:
    """
    Ask the summary LLM to find and fix Whisper mishearings in *transcript*,
    using *brief* and *detailed* summaries as a trusted factual reference.

    The LLM is prompted to return a JSON object with a single ``replacements``
    list of ``{"wrong": "...", "right": "..."}`` pairs.  Each pair is applied
    as a whole-word, case-insensitive substitution.

    Fail-safe: if the LLM call fails or returns malformed JSON the original
    transcript is returned unchanged so ingestion is never blocked.
    """
    import ollama as _ollama

    prompt = cfg.transcript_correction_prompt.format(
        brief=brief,
        detailed=detailed,
        transcript=transcript,
    )

    try:
        client = _ollama.AsyncClient(host=cfg.ollama_url)
        response = await client.chat(
            model=cfg.summary_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        raw: str = response.message.content
    except Exception as exc:
        logger.warning(
            "correct_transcript: LLM call failed (%s) — using original transcript.", exc
        )
        return transcript

    # Strip markdown code fences if the model wrapped its JSON.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
        replacements: list[dict] = data.get("replacements", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "correct_transcript: could not parse LLM response as JSON (%s) — using original.",
            exc,
        )
        logger.debug("Raw LLM response: %s", raw)
        return transcript

    if not replacements:
        logger.info("  Transcript correction: no replacements suggested.")
        return transcript

    # ── Critique: filter out over-corrections before applying ─────────────────
    replacements = await _critique_replacements(
        replacements, transcript, brief, detailed, cfg, logger
    )

    if not replacements:
        logger.info("  Transcript correction: all suggestions rejected by critique — no changes made.")
        return transcript

    corrected = transcript
    for item in replacements:
        wrong = item.get("wrong", "")
        right = item.get("right", "")
        if not wrong or not right:
            continue
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        corrected, n = pattern.subn(right, corrected)
        if n:
            logger.info(
                "  Transcript correction: '%s' → '%s' (%d occurrence(s))", wrong, right, n
            )

    return corrected


async def _critique_replacements(
    replacements: list[dict],
    transcript: str,
    brief: str,
    detailed: str,
    cfg: Config,
    logger: logging.Logger,
) -> list[dict]:
    """
    Ask the summary LLM to review the proposed replacement list and filter out
    any over-corrections — e.g. replacing a legitimate technical term with an
    incorrect one, or making changes that contradict the reference summaries.

    Returns the approved subset of *replacements*.  On any failure the original
    list is returned unchanged (fail-safe).
    """
    import ollama as _ollama

    prompt = cfg.transcript_critique_prompt.format(
        brief=brief,
        detailed=detailed,
        transcript=transcript,
        replacements=json.dumps(replacements, ensure_ascii=False, indent=2),
    )

    try:
        client = _ollama.AsyncClient(host=cfg.ollama_url)
        response = await client.chat(
            model=cfg.summary_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        raw: str = response.message.content
    except Exception as exc:
        logger.warning(
            "_critique_replacements: LLM call failed (%s) — using original replacements.", exc
        )
        return replacements

    # Strip markdown code fences if the model wrapped its JSON.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
        approved: list[dict] = data.get("approved", [])
        rejected: list[dict] = data.get("rejected", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "_critique_replacements: could not parse critique response as JSON (%s)"
            " — using original replacements.",
            exc,
        )
        logger.debug("Raw critique response: %s", raw)
        return replacements

    for item in rejected:
        logger.info(
            "  Transcript critique REJECTED: '%s' → '%s' — %s",
            item.get("wrong", "?"),
            item.get("right", "?"),
            item.get("reason", "no reason given"),
        )

    return approved


def detect_faulty_docs(docs_dir: Path, state: StateDB) -> list[str]:
    """
    Scan every saved knowledge-document in *docs_dir* for known quality issues:
    - CJK (Chinese) characters — LLM hallucination artefact
    - Known Whisper mishearings (see ``_KNOWN_BAD_TERMS``)

    Returns a list of video *filenames* (e.g. 'cisco23.mp4') whose saved
    ``_ingested.txt`` is faulty.  Only files tracked in the state DB as
    'ingested' are considered.
    """
    faulty: list[str] = []

    if not docs_dir.exists():
        return faulty

    # Build a quick lookup: stem → filename for all ingested rows.
    all_rows = state.get_all()
    stem_to_filename = {
        Path(r["filename"]).stem: r["filename"]
        for r in all_rows
        if r["status"] == "ingested"
    }

    for doc_path in sorted(docs_dir.glob("*_ingested.txt")):
        # Strip the "_ingested" suffix to recover the video stem.
        stem = doc_path.stem[: -len("_ingested")]
        if stem not in stem_to_filename:
            continue  # not in DB (orphan file), skip

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue

        if _is_faulty(content):
            faulty.append(stem_to_filename[stem])

    return faulty
