#!/usr/ایی/env python3
import json
from pathlib import Path
from collections import defaultdict

def main():
    kv_path = Path("knowledge_db/kv_store_full_docs.json")
    if not kv_path.exists():
        print(f"File not found: {kv_path.absolute()}")
        return

    with open(kv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # title -> list of doc_ids
    title_to_ids = defaultdict(list)
    
    for doc_id, entry in data.items():
        content = entry if isinstance(entry, str) else entry.get("content", "")
        # The title is the first line, e.g. "# Cisco09"
        first_line = content.split("\n")[0].strip()
        if first_line.startswith("# "):
            title = first_line[2:]
        else:
            title = "UNKNOWN_TITLE"
        
        title_to_ids[title].append(doc_id)

    duplicates = 0
    orphans = 0
    clean = 0
    unknown = 0

    print("=" * 60)
    print("LightRAG DB State Verification")
    print("=" * 60)
    
    for title, ids in sorted(title_to_ids.items()):
        if title == "UNKNOWN_TITLE":
            unknown += 1
            continue
            
        if len(ids) > 1:
            print(f"[DUPLICATES] {title:<15} -> {ids}")
            duplicates += 1
        else:
            doc_id = ids[0]
            # If the ID doesn't match the title (ignoring case) and looks like a hash
            if doc_id.lower() != title.lower():
                print(f"[ORPHAN]     {title:<15} -> {ids}")
                orphans += 1
            else:
                clean += 1

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Unique Videos   : {len(title_to_ids) - (1 if unknown else 0)}")
    print(f"Clean (1 stem ID)     : {clean}")
    print(f"Orphans (1 hash ID)   : {orphans}")
    print(f"Duplicates (>1 ID)    : {duplicates}")
    if unknown:
        print(f"Unknown Titles        : {unknown}")

if __name__ == "__main__":
    main()
