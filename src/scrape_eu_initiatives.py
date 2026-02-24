"""
Scrape all 1821 EU "Have Your Say" initiatives from the Better Regulation API.

Source listing:
https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en
  ?feedbackOpenDateFrom=01-12-2019&feedbackOpenDateClosedBy=30-11-2024

API endpoint:
https://ec.europa.eu/info/law/better-regulation/brpapi/searchInitiatives
"""

import csv
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL = (
    "https://ec.europa.eu/info/law/better-regulation/brpapi/searchInitiatives"
    "?feedbackOpenDateFrom=2019/12/01"
    "&feedbackOpenDateClosedBy=2024/11/30"
    "&language=EN"
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

    Returns the inner page object (with content, totalPages, etc.).
    """
    url = f"{API_URL}&page={page}&size={PAGE_SIZE}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["initiativeResultDtoPage"]
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1} after error: {exc}  (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def extract_initiatives(data: dict) -> list[dict]:
    """Extract initiative records from a page response."""
    records = []
    for item in data.get("content", []):
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

        records.append({
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
        })
    return records


def main():
    out_path = Path(__file__).parent.parent / "eu_initiatives.csv"
    all_records: list[dict] = []

    # Fetch first page to get total count
    print("Fetching page 1 ...")
    first = fetch_page(0)
    total_pages = first["totalPages"]
    total_elements = first["totalElements"]
    print(f"Total initiatives: {total_elements}  |  Pages: {total_pages}")

    all_records.extend(extract_initiatives(first))

    for page in range(1, total_pages):
        print(f"Fetching page {page + 1}/{total_pages} ...")
        data = fetch_page(page)
        all_records.extend(extract_initiatives(data))
        time.sleep(0.3)  # be polite

    print(f"\nCollected {len(all_records)} initiatives")

    # Write CSV
    fieldnames = [
        "id", "reference", "short_title", "initiative_status", "act_type",
        "feedback_status", "feedback_start", "feedback_end", "topics", "url",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
