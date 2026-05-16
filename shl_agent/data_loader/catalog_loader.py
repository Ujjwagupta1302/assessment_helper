"""
catalog_loader.py
==================
Loads the SHL product catalog JSON once at startup and exposes:

- CATALOG: the full list of items (for reference)
- URL_TO_ITEM: a {url -> item} lookup for fast URL validation
- get_catalog_for_prompt(): returns a compact catalog string for the
  Gemini selection prompt (only the fields the model actually needs).

The catalog file is loaded with strict=False because the source JSON
sometimes contains stray control characters inside description fields.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


CATALOG_FILE = Path(__file__).parent.parent.parent / "input_data" / "data.json"


def _load_catalog(path: Path) -> List[Dict]:
    """Load the catalog JSON, tolerating control characters."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return json.loads(raw, strict=False)


CATALOG: List[Dict] = _load_catalog(CATALOG_FILE)


URL_TO_ITEM: Dict[str, Dict] = {
    item["link"]: item for item in CATALOG if item.get("link")
}


def get_item_by_url(url: str) -> Optional[Dict]:
    """Return the catalog item for a URL, or None if the URL is unknown."""
    return URL_TO_ITEM.get(url)


def is_valid_url(url: str) -> bool:
    """True iff this URL belongs to a real catalog item."""
    return url in URL_TO_ITEM


def _compact_item(item: Dict) -> Dict:
    """
    Return a slimmed-down version of a catalog item suitable for prompts.
    Strips the heavy '_raw' fields, scrape timestamps, and status flags.
    """
    return {
        "entity_id": item.get("entity_id"),
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "description": item.get("description", ""),
        "keys": item.get("keys", []),
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
    }


def get_catalog_for_prompt() -> str:
    """
    Return the entire catalog as a JSON string for inclusion in the
    Gemini selection prompt. Only essential fields are included.
    """
    compact = [_compact_item(item) for item in CATALOG]
    return json.dumps(compact, ensure_ascii=False, indent=None)


def get_filtered_catalog_for_prompt(urls: List[str]) -> str:
    """
    Return a subset of the catalog as a JSON string, in the order of
    the URLs provided. Unknown URLs are skipped silently.

    Used by the agent to send Gemini only the items selected by the
    vector search, instead of all 377 catalog items.
    """
    items = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        item = URL_TO_ITEM.get(url)
        if item is not None:
            items.append(_compact_item(item))
    return json.dumps(items, ensure_ascii=False, indent=None)


def get_catalog_summary() -> Dict:
    """Quick diagnostic summary, useful at startup to verify load."""
    keys_set = set()
    for item in CATALOG:
        for k in item.get("keys", []):
            keys_set.add(k)
    return {
        "total_items": len(CATALOG),
        "unique_urls": len(URL_TO_ITEM),
        "unique_keys": sorted(keys_set),
    }


if __name__ == "__main__":
    summary = get_catalog_summary()
    print(f"Loaded {summary['total_items']} items")
    print(f"Unique URLs: {summary['unique_urls']}")
    print(f"Unique keys ({len(summary['unique_keys'])}):")
    for k in summary['unique_keys']:
        print(f"  - {k}")

    prompt_str = get_catalog_for_prompt()
    print(f"\nCompact catalog string length: {len(prompt_str):,} chars")
    print(f"Estimated tokens: {len(prompt_str) // 4:,}")