#!/usr/bin/env python3
"""
LectoGraph Discovery Tool — find potential Whisper hallucinations across the corpus.

Logic:
1. Scan all *_ingested.txt files in the docs/ directory.
2. Extract all-caps acronyms and technical-looking words.
3. Perform frequency analysis: rare terms are likely mishearings.
4. (Optional) Audit the rare terms using the project's summary LLM.
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

# regex for acronyms (2-8 chars) and common technical patterns
ACRONYM_PAT = re.compile(r"\b[A-ZÄÅÖ]{2,8}\b")
# Regex for words that look like config keys or commands (e.g. eth0, ipconfig)
TECH_PAT = re.compile(r"\b[a-zA-Z0-9]{3,15}[0-9]+[a-zA-Z0-9]*\b")

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
            counts.update(acronyms)
            
            # Find all alphanumeric tech patterns
            tech_terms = TECH_PAT.findall(text)
            counts.update(tech_terms)
        except Exception as e:
            print(f"Error reading {doc_path.name}: {e}")
            
    return counts

async def audit_with_llm(rare_terms: List[str], ollama_url: str, model: str):
    """Ask the LLM to identify which rare terms are likely hallucinations."""
    import ollama
    
    print(f"Auditing {len(rare_terms)} rare terms with {model}...")
    
    prompt = (
        "Du är en IT-expert. Nedan är en lista på termer funna i transkriptioner "
        "av IT-lektioner. De flesta är korrekta, men vissa är felavlyssningar (hallucinationer) "
        "från en AI (Whisper).\n\n"
        "Din uppgift: Identifiera vilka som ser ut som felaktiga avlyssningar och föreslå "
        "vad de egentligen borde vara (t.ex. 'DOCP' -> 'DHCP').\n\n"
        "Returnera resultatet som ett JSON-objekt med fältet 'hallucinations' som är en "
        "lista av {\"wrong\": \"...\", \"right\": \"...\"} objekt.\n\n"
        f"Termer att granska: {', '.join(rare_terms)}\n\n"
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
        # Basic markdown cleaning
        content = re.sub(r"```[a-z]*\n?", "", content).replace("```", "").strip()
        return json.loads(content).get("hallucinations", [])
    except Exception as e:
        print(f"LLM Audit failed: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Discover potential Whisper hallucinations.")
    parser.add_argument("--docs", default="./docs", help="Path to docs directory")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum frequency to ignore (default: 1)")
    parser.add_argument("--max-freq", type=int, default=3, help="Maximum frequency to be considered 'rare' (default: 3)")
    parser.add_argument("--audit", action="store_true", help="Audit rare terms with Ollama")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (for audit settings)")
    
    args = parser.parse_args()
    docs_dir = Path(args.docs)
    
    if not docs_dir.exists():
        print(f"Error: Directory not found: {docs_dir}")
        return

    counts = discover_candidates(docs_dir)
    
    # Filter for rare terms
    rare = [term for term, count in counts.items() if args.min_freq <= count <= args.max_freq]
    rare.sort()

    print(f"\nFound {len(rare)} rare terms (appearing {args.min_freq}-{args.max_freq} times):")
    for term in rare:
        print(f"  {term:<15} ({counts[term]} hits)")

    if args.audit and rare:
        import asyncio
        import yaml
        
        # Load settings from config.yaml
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: Config not found: {cfg_path}")
            return
            
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        ollama_url = cfg.get("ollama_url", "http://127.0.0.1:11434")
        model = cfg.get("summary_model", "gemma4:31b")
        
        hallucinations = asyncio.run(audit_with_llm(rare, ollama_url, model))
        
        if hallucinations:
            print("\nPotential Hallucinations discovered by LLM:")
            print("-" * 40)
            for h in hallucinations:
                print(f"  {h.get('wrong', '?'):<15} -> {h.get('right', '?')}")
            print("-" * 40)
            print(f"\nTo fix these, add them to _KNOWN_BAD_TERMS in pipeline.py.")
        else:
            print("\nLLM found no obvious hallucinations in the rare terms list.")

if __name__ == "__main__":
    main()
