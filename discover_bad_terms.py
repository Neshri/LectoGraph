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
    
    # Smaller batches for better feedback and less model strain
    batch_size = 20
    all_hallucinations = []
    total_batches = (len(candidates) + batch_size - 1) // batch_size
    
    print(f"Auditing {len(candidates)} near-miss pairs with {model} in {total_batches} batches...")
    
    for i in range(0, len(candidates), batch_size):
        current_batch = (i // batch_size) + 1
        print(f"  -> Processing batch {current_batch}/{total_batches}...", end="", flush=True)
        
        batch = candidates[i:i+batch_size]
        pairs_str = "\n".join([f"- {wrong} vs {right}" for wrong, right in batch])
        
        prompt = (
            "Du är en IT-expert som granskar transkriptionsfel.\n"
            "Jag har par av ord (Kandidat vs Mål). Vissa är två OLIKA giltiga IT-termer "
            "(t.ex. 'ACPI' vs 'API'), medan andra är felhörningar (t.ex. 'DOCP' vs 'DHCP').\n\n"
            "Din uppgift:\n"
            "1. Om BÅDA orden är legitima och distinkta IT-termer, ignorera paret.\n"
            "2. Om Kandidaten ser ut som en felstavning/felhörning av Målet, lägg till det i listan.\n"
            "3. Fokusera särskilt på ord som låter lika fonetiskt.\n\n"
            f"Par att granska:\n{pairs_str}\n\n"
            "Svara ENDAST med JSON i formatet: {\"hallucinations\": [{\"wrong\": \"...\", \"right\": \"...\"}]}"
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
            res = json.loads(content).get("hallucinations", [])
            all_hallucinations.extend(res)
            print(" Done.")
        except Exception as e:
            print(f" Failed: {e}")
            
    return all_hallucinations

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
    print("Searching for near-misses (Length >= 4, Ratio >= 1.5x)...")

    near_misses = []
    for r in rare:
        if len(r) < 4:
            continue
            
        r_count = counts[r]
        for c in common:
            if abs(len(r) - len(c)) > 1:
                continue
                
            c_count = counts[c]
            # Lower ratio (1.5x) to catch persistent hallucinations like DOCP
            if c_count < r_count * 1.5:
                continue
                
            if levenshtein(r, c) == 1:
                near_misses.append((r, c, r_count, c_count))
                break 

    if not near_misses:
        print("No candidates found.")
        return

    print(f"\nFound {len(near_misses)} candidates.")

    if args.audit:
        import asyncio
        import yaml
        cfg_path = Path(args.config)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        ollama_url = cfg.get("ollama_url", "http://127.0.0.1:11434")
        model = cfg.get("summary_model", "gemma4:31b")
        
        # Prepare pairs for the LLM
        audit_pairs = [(m[0], m[1]) for m in near_misses]
        hallucinations = asyncio.run(audit_with_llm(audit_pairs, ollama_url, model))
        
        if hallucinations:
            print("\n" + "="*60)
            print(f"{'WHISPER ERROR':<20} -> {'ACTUAL TERM':<20} | {'REASON'}")
            print("-" * 60)
            for h in hallucinations:
                wrong = h.get('wrong', '?')
                right = h.get('right', '?')
                # Find counts from our near_misses list
                stats = next((m for m in near_misses if m[0] == wrong), (0,0,0,0))
                print(f"  {wrong:<18} -> {right:<18} | {stats[2]} vs {stats[3]} hits")
            print("=" * 60)
            print(f"\nAdd the verified errors to _KNOWN_BAD_TERMS in pipeline.py.")
        else:
            print("\nLLM found no obvious hallucinations in the candidate list.")

if __name__ == "__main__":
    main()
