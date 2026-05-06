#!/usr/bin/env python3
"""
LectoGraph Discovery Tool — find potential Whisper hallucinations across the corpus.

Patched version:
- Columnar output to prevent CLI flooding.
- Noise filtering for digits and short terms.
- Support for saving results to a file.
"""

from __future__ import annotations

import re
import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

# regex for acronyms (2-8 chars) and common technical patterns
ACRONYM_PAT = re.compile(r"\b[A-ZÄÅÖ]{2,8}\b")
# Regex for words that look like config keys or commands (e.g. eth0, ipconfig)
TECH_PAT = re.compile(r"\b[a-zA-Z0-9]{3,15}[0-9]+[a-zA-Z0-9]*\b")

def is_noise(term: str) -> bool:
    """Filter out obviously non-technical noise."""
    # Too short or just numbers
    if len(term) < 2 or term.isdigit():
        return True
    # Mostly digits (e.g. 1000, 2024)
    if sum(c.isdigit() for c in term) / len(term) > 0.5:
        return True
    return False

def print_columns(items: List[str], cols: int = 4, width: int = 25):
    """Print a list of strings in balanced columns."""
    for i in range(0, len(items), cols):
        chunk = items[i : i + cols]
        line = "".join(f"{item:<{width}}" for item in chunk)
        print(line)

def discover_candidates(docs_dir: Path) -> Counter:
    """Scan all docs and return a frequency map of technical-looking terms."""
    counts = Counter()
    
    doc_files = list(docs_dir.glob("*_ingested.txt"))
    print(f"Scanning {len(doc_files)} documents in {docs_dir}...")

    for doc_path in doc_files:
        try:
            text = doc_path.read_text(encoding="utf-8")
            # Find all all-caps acronyms
            acronyms = ACRONYM_PAT.findall(text)
            counts.update([a for a in acronyms if not is_noise(a)])
            
            # Find all alphanumeric tech patterns
            tech_terms = TECH_PAT.findall(text)
            counts.update([t for t in tech_terms if not is_noise(t)])
        except Exception as e:
            print(f"Error reading {doc_path.name}: {e}")
            
    return counts

async def audit_with_llm(rare_terms: List[str], ollama_url: str, model: str):
    """Ask the LLM to identify which rare terms are likely hallucinations."""
    import ollama
    
    # Limit audit to top 50 to avoid timeout/cost
    audit_subset = rare_terms[:50]
    print(f"Auditing {len(audit_subset)} most suspicious terms with {model}...")
    
    prompt = (
        "Du är en IT-expert. Nedan är en lista på termer funna i transkriptioner "
        "av IT-lektioner. De flesta är korrekta, men vissa är felavlyssningar (hallucinationer) "
        "från en AI (Whisper).\n\n"
        "Din uppgift: Identifiera vilka som ser ut som felaktiga avlyssningar och föreslå "
        "vad de egentligen borde vara (t.ex. 'DOCP' -> 'DHCP').\n\n"
        "Returnera resultatet som ett JSON-objekt med fältet 'hallucinations' som är en "
        "lista av {\"wrong\": \"...\", \"right\": \"...\"} objekt.\n\n"
        f"Termer att granska: {', '.join(audit_subset)}\n\n"
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
    parser.add_argument("--min-freq", type=int, default=1, help="Min freq to consider (default: 1)")
    parser.add_argument("--max-freq", type=int, default=3, help="Max freq to consider (default: 3)")
    parser.add_argument("--audit", action="store_true", help="Audit rare terms with Ollama")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", help="Save results to this file")
    
    args = parser.parse_args()
    docs_dir = Path(args.docs)
    
    if not docs_dir.exists():
        print(f"Error: Directory not found: {docs_dir}")
        return

    counts = discover_candidates(docs_dir)
    
    # Filter for rare terms, sorted by frequency (ascending)
    rare_items = sorted(
        [(term, count) for term, count in counts.items() if args.min_freq <= count <= args.max_freq],
        key=lambda x: x[1]
    )
    
    if not rare_items:
        print("No rare terms found.")
        return

    print(f"\nFound {len(rare_items)} rare terms (appearing {args.min_freq}-{args.max_freq} times):")
    print("-" * 100)
    display_list = [f"{t} ({c})" for t, c in rare_items]
    print_columns(display_list)
    print("-" * 100)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for term, count in rare_items:
                f.write(f"{term}\t{count}\n")
        print(f"Results saved to {out_path}")

    if args.audit:
        import asyncio
        import yaml
        
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: Config not found: {cfg_path}")
            return
            
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        ollama_url = cfg.get("ollama_url", "http://127.0.0.1:11434")
        model = cfg.get("summary_model", "gemma4:31b")
        
        # Sort by frequency for audit too
        rare_terms = [t for t, c in rare_items]
        hallucinations = asyncio.run(audit_with_llm(rare_terms, ollama_url, model))
        
        if hallucinations:
            print("\nPotential Hallucinations discovered by LLM:")
            print("=" * 60)
            for h in hallucinations:
                print(f"  {h.get('wrong', '?'):<20} -> {h.get('right', '?')}")
            print("=" * 60)
            print(f"\nTo fix these, add them to _KNOWN_BAD_TERMS in pipeline.py.")
        else:
            print("\nLLM found no obvious hallucinations in the rare terms list.")

if __name__ == "__main__":
    main()
