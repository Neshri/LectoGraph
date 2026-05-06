#!/usr/bin/env python3
"""
LectoGraph Discovery Tool — find potential Whisper hallucinations across the corpus.

Advanced version:
- Identifies "Common Terms" in the corpus.
- Uses Levenshtein distance to find "Rare Terms" that are near-misses of common terms.
- Filters out document IDs (Win123, Linux45, etc.).
"""

from __future__ import annotations

import re
import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

# regex for acronyms (2-8 chars) and common technical patterns
ACRONYM_PAT = re.compile(r"\b[A-ZÄÅÖ]{2,8}\b")
TECH_PAT = re.compile(r"\b[a-zA-Z0-9]{3,15}[0-9]+[a-zA-Z0-9]*\b")

# Pattern for common auto-generated IDs in this project
ID_PAT = re.compile(r"^(Win|Linux|Cisco|Pc|Misc|Network|Ai|test|web|klient|server|DATOR)[0-9]+$", re.IGNORECASE)

# Common Swedish/English words to ignore
STOP_WORDS = {"ALLA", "PÅGÅR", "TILL", "ELLER", "FALSK", "TUM", "FRÅN", "OCH", "DET", "SOM", "ATT", "DEN", "VAR", "MED", "MEN"}

def levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance implementation."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def is_noise(term: str) -> bool:
    if len(term) < 2 or term.isdigit():
        return True
    if sum(c.isdigit() for c in term) / len(term) > 0.5:
        return True
    if ID_PAT.match(term):
        return True
    if term.upper() in STOP_WORDS:
        return True
    return False

def discover_candidates(docs_dir: Path) -> Counter:
    counts = Counter()
    doc_files = list(docs_dir.glob("*_ingested.txt"))
    print(f"Scanning {len(doc_files)} documents...")

    for doc_path in doc_files:
        try:
            text = doc_path.read_text(encoding="utf-8")
            acronyms = ACRONYM_PAT.findall(text)
            counts.update([a for a in acronyms if not is_noise(a)])
            tech_terms = TECH_PAT.findall(text)
            counts.update([t for t in tech_terms if not is_noise(t)])
        except Exception as e:
            print(f"Error reading {doc_path.name}: {e}")
    return counts

async def audit_with_llm(candidates: List[Tuple[str, str]], ollama_url: str, model: str):
    import ollama
    print(f"Auditing {len(candidates)} near-miss pairs with {model}...")
    
    pairs_str = "\n".join([f"- {wrong} (liknar {right})" for wrong, right in candidates])
    
    prompt = (
        "Du är en IT-expert. Jag har hittat ord i transkriptioner som verkar vara "
        "felavlyssningar (hallucinationer) eftersom de liknar kända IT-begrepp.\n\n"
        "Din uppgift: Avgör vilka som är faktiska felhörningar och bekräfta rättningen.\n\n"
        "Returnera resultatet som ett JSON-objekt med fältet 'hallucinations' som är en "
        "lista av {\"wrong\": \"...\", \"right\": \"...\"} objekt.\n\n"
        f"Kandidater:\n{pairs_str}\n\n"
        "Svara ENDAST med JSON."
    )

    try:
        client = ollama.AsyncClient(host=ollama_url)
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        content = response.message.content
        content = re.sub(r"```[a-z]*\n?", "", content).replace("```", "").strip()
        return json.loads(content).get("hallucinations", [])
    except Exception as e:
        print(f"LLM Audit failed: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Discover potential Whisper hallucinations.")
    parser.add_argument("--docs", default="./docs", help="Path to docs directory")
    parser.add_argument("--threshold", type=int, default=10, help="Min freq to be considered 'Common'")
    parser.add_argument("--max-rare", type=int, default=3, help="Max freq to be considered 'Rare'")
    parser.add_argument("--audit", action="store_true", help="Audit near-misses with Ollama")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    
    args = parser.parse_args()
    docs_dir = Path(args.docs)
    
    if not docs_dir.exists():
        print(f"Error: Directory not found: {docs_dir}")
        return

    counts = discover_candidates(docs_dir)
    
    common = sorted([t for t, c in counts.items() if c >= args.threshold])
    rare = sorted([t for t, c in counts.items() if 1 <= c <= args.max_rare])

    print(f"Found {len(common)} common terms and {len(rare)} rare terms.")
    print("Searching for high-confidence near-misses (Length >= 4, Ratio >= 20x)...")

    near_misses = []
    for r in rare:
        if len(r) < 4: # Skip short noisy acronyms
            continue
            
        r_count = counts[r]
        for c in common:
            if abs(len(r) - len(c)) > 1:
                continue
                
            c_count = counts[c]
            # Only care if the common term is significantly more frequent
            if c_count < r_count * 20:
                continue
                
            if levenshtein(r, c) == 1:
                near_misses.append((r, c, r_count, c_count))
                break 

    if not near_misses:
        print("No high-confidence near-misses found.")
        return

    print(f"\nFound {len(near_misses)} high-confidence hallucinations:")
    print("-" * 80)
    print(f"{'Rare Term':<20} {'Freq':<5} -> {'Common Target':<20} {'Freq':<5}")
    print("-" * 80)
    for wrong, right, r_c, c_c in near_misses:
        print(f"  {wrong:<18} ({r_c})    -> {right:<18} ({c_c})")
    print("-" * 80)

    if args.audit:
        import asyncio
        import yaml
        cfg_path = Path(args.config)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        ollama_url = cfg.get("ollama_url", "http://127.0.0.1:11434")
        model = cfg.get("summary_model", "gemma4:31b")
        
        hallucinations = asyncio.run(audit_with_llm(near_misses, ollama_url, model))
        
        if hallucinations:
            print("\nLLM Confirmed Hallucinations:")
            print("=" * 60)
            for h in hallucinations:
                print(f"  {h.get('wrong', '?'):<15} -> {h.get('right', '?')}")
            print("=" * 60)
            print(f"\nAdd these to _KNOWN_BAD_TERMS in pipeline.py.")

if __name__ == "__main__":
    main()
