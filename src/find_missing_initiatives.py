"""Find initiative IDs that are missing or have incomplete data in eu_initiative_details.jsonl.

Reports:
  - IDs in CSV but not in JSONL at all
  - IDs in JSONL with publications that have feedback_error (feedback API returned 400)
"""

import csv
import json
from pathlib import Path


def main():
    base = Path(__file__).parent.parent
    csv_path = base / "eu_initiatives.csv"
    jsonl_path = base / "eu_initiative_details.jsonl"

    with open(csv_path, encoding="utf-8") as f:
        csv_ids = {int(row["id"]) for row in csv.DictReader(f)}

    jsonl_ids = set()
    fb_error_initiatives = {}  # id -> (title, list of (pub_id, pub_type, error))
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            jsonl_ids.add(obj["id"])
            for pub in obj.get("publications", []):
                if "feedback_error" in pub:
                    entry = fb_error_initiatives.setdefault(
                        obj["id"], (obj.get("short_title", ""), [])
                    )
                    entry[1].append(
                        (pub["publication_id"], pub["type"], pub["feedback_error"])
                    )

    missing = sorted(csv_ids - jsonl_ids)

    print(f"CSV unique IDs: {len(csv_ids)}")
    print(f"JSONL unique IDs: {len(jsonl_ids)}")

    print(f"\nMissing from JSONL entirely: {len(missing)}")
    for mid in missing:
        print(f"  {mid}")

    print(f"\nInitiatives with feedback errors: {len(fb_error_initiatives)}")
    total_fb_errors = 0
    for init_id in sorted(fb_error_initiatives):
        title, errs = fb_error_initiatives[init_id]
        total_fb_errors += len(errs)
        details = ", ".join(f"{pub_id} ({pub_type})" for pub_id, pub_type, _ in errs)
        base_url = f"https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/{init_id}"
        print(f"  {init_id}: {title}")
        print(f"    {base_url}")
        print(f"    {base_url}/public-consultation")
        print(f"    {len(errs)} failed publication(s) — {details}")

    print(f"\nTotal: {len(missing)} missing + {len(fb_error_initiatives)} with feedback errors "
          f"({total_fb_errors} failed publications)")


if __name__ == "__main__":
    main()
