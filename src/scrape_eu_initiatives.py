"""
Scrape all EU "Have Your Say" initiatives from the Better Regulation API.

Source listing:
https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en

API endpoint:
https://ec.europa.eu/info/law/better-regulation/brpapi/searchInitiatives

Caches raw API page responses to a directory for resume/offline use.
Produces eu_initiatives.csv and eu_initiatives_raw.json.
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL = (
    "https://ec.europa.eu/info/law/better-regulation/brpapi/searchInitiatives"
    "?language=EN"
)
PAGE_SIZE = 10
BASE_INITIATIVE_URL = (
    "https://ec.europa.eu/info/law/better-regulation"
    "/have-your-say/initiatives"
)


def slugify(text: str) -> str:
    """Convert a title into a URL-friendly slug matching the EU site convention."""
    text = re.sub(r"[^\w\s-]", "", text)  # strip non-alphanumeric (keep hyphens)
    text = re.sub(r"[\s_]+", "-", text.strip())  # spaces/underscores to hyphens
    return text


def fetch_page(page: int) -> dict:
    """Fetch a single page of results from the API.

    Returns the full API response (with initiativeResultDtoPage).
    """
    url = f"{API_URL}&page={page}&size={PAGE_SIZE}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1} after error: {exc}  (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def extract_initiative_record(item: dict) -> dict:
    """Extract a flat CSV-friendly record from a raw initiative item."""
    init_id = int(item["id"])
    short_title = item.get("shortTitle", "")
    reference = item.get("reference", "")
    status = item.get("initiativeStatus", "")
    act_type = item.get("foreseenActType", "")

    # Feedback status from currentStatuses
    feedback_status = ""
    feedback_start = ""
    feedback_end = ""
    for cs in item.get("currentStatuses", []):
        if cs.get("isCurrent"):
            feedback_status = cs.get("receivingFeedbackStatus", "")
            feedback_start = cs.get("feedbackStartDate", "")
            feedback_end = cs.get("feedbackEndDate", "")
            break

    # Topics
    topics = "; ".join(t.get("label", "") for t in item.get("topics", []))

    # Construct page URL
    slug = slugify(short_title)
    page_url = f"{BASE_INITIATIVE_URL}/{init_id}-{slug}_en"

    return {
        "id": init_id,
        "reference": reference,
        "short_title": short_title,
        "initiative_status": status,
        "act_type": act_type,
        "feedback_status": feedback_status,
        "feedback_start": feedback_start,
        "feedback_end": feedback_end,
        "topics": topics,
        "url": page_url,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all EU 'Have Your Say' initiatives."
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory to cache raw API page responses (default: eu_initiatives_cache/).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output CSV path (default: eu_initiatives.csv).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    out_csv = Path(args.output) if args.output else base_dir / "eu_initiatives.csv"
    out_json = out_csv.with_suffix(".json").with_name(
        out_csv.stem + "_raw.json"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else base_dir / "eu_initiatives_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch first page to get total count (or load from cache)
    page0_cache = cache_dir / "page_0000.json"
    if page0_cache.is_file():
        print("Loading page 1 from cache...")
        with open(page0_cache, encoding="utf-8") as f:
            first_response = json.load(f)
    else:
        print("Fetching page 1 ...")
        first_response = fetch_page(0)
        with open(page0_cache, "w", encoding="utf-8") as f:
            json.dump(first_response, f, ensure_ascii=False, indent=2)

    first = first_response["initiativeResultDtoPage"]
    total_pages = first["totalPages"]
    total_elements = first["totalElements"]
    print(f"Total initiatives: {total_elements}  |  Pages: {total_pages}")

    # Collect all raw initiative items from all pages
    all_raw_items = list(first.get("content", []))

    for page in range(1, total_pages):
        page_cache = cache_dir / f"page_{page:04d}.json"
        if page_cache.is_file():
            with open(page_cache, encoding="utf-8") as f:
                response = json.load(f)
            print(f"Loaded page {page + 1}/{total_pages} from cache ({len(response['initiativeResultDtoPage'].get('content', []))} items)")
        else:
            print(f"Fetching page {page + 1}/{total_pages} ...")
            response = fetch_page(page)
            with open(page_cache, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            time.sleep(0.3)  # be polite

        all_raw_items.extend(response["initiativeResultDtoPage"].get("content", []))

    print(f"\nCollected {len(all_raw_items)} initiatives")

    # Write raw JSON (full API data for each initiative)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_raw_items, f, ensure_ascii=False, indent=2)
    print(f"Raw JSON: {out_json}")

    # Extract CSV records
    csv_records = [extract_initiative_record(item) for item in all_raw_items]

    fieldnames = [
        "id", "reference", "short_title", "initiative_status", "act_type",
        "feedback_status", "feedback_start", "feedback_end", "topics", "url",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_records)

    print(f"CSV: {out_csv}")
    print(f"Cache: {cache_dir}/ ({total_pages} page files)")


if __name__ == "__main__":
    main()
