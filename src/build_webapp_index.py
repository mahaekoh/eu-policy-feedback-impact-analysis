"""Pre-compute the webapp initiative index.

Reads all initiative detail JSONs from data/scrape/initiative_details/,
extracts metadata, computes feedback statistics, deduplicates initiatives
sharing identical feedback IDs, and writes a single JSON index file for
the webapp to consume.

Usage:
    python3 src/build_webapp_index.py data/scrape/initiative_details/ -o data/webapp/initiative_index.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

TIMELINE_BUCKETS = 20


def parse_date(date_str):
    """Parse 'YYYY/MM/DD HH:MM:SS' or ISO 8601 to a datetime (UTC)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_summary(data):
    """Build an InitiativeSummary dict from a parsed initiative JSON."""
    init_id = data.get("id")
    if not init_id:
        return None

    publications = data.get("publications", [])

    # Sum total_feedback and check for open feedback
    total_feedback = 0
    has_open_feedback = False
    for pub in publications:
        total_feedback += pub.get("total_feedback", 0)
        if pub.get("feedback_status") == "OPEN":
            has_open_feedback = True

    # Collect feedback stats
    feedback_ids = []
    country_counts = {}
    user_type_counts = {}
    feedback_dates = []

    for pub in publications:
        for fb in pub.get("feedback", []):
            fb_id = fb.get("id")
            if fb_id is not None:
                feedback_ids.append(fb_id)

            country = fb.get("country")
            if country:
                country_counts[country] = country_counts.get(country, 0) + 1

            user_type = fb.get("user_type")
            if user_type:
                user_type_counts[user_type] = (
                    user_type_counts.get(user_type, 0) + 1
                )

            date = fb.get("date")
            if date:
                dt = parse_date(date)
                if dt:
                    feedback_dates.append(dt)

    feedback_ids.sort()

    # Compute feedback timeline histogram
    published_date = data.get("published_date", "")
    start_dt = parse_date(published_date)
    feedback_timeline = []
    last_feedback_date = ""

    if feedback_dates:
        end_dt = max(feedback_dates)
        last_feedback_date = end_dt.isoformat()

        if start_dt and end_dt > start_dt:
            start_ms = start_dt.timestamp()
            end_ms = end_dt.timestamp()
            bucket_width = (end_ms - start_ms) / TIMELINE_BUCKETS
            feedback_timeline = [0] * TIMELINE_BUCKETS
            for dt in feedback_dates:
                idx = min(
                    int((dt.timestamp() - start_ms) / bucket_width),
                    TIMELINE_BUCKETS - 1,
                )
                if idx >= 0:
                    feedback_timeline[idx] += 1

    return {
        "id": init_id,
        "short_title": data.get("short_title", ""),
        "department": data.get("department", ""),
        "stage": data.get("stage", ""),
        "status": data.get("status", ""),
        "topics": data.get("topics", []),
        "policy_areas": data.get("policy_areas", []),
        "published_date": published_date,
        "last_cached_at": data.get("last_cached_at", ""),
        "type_of_act": data.get("type_of_act", ""),
        "reference": data.get("reference", ""),
        "total_feedback": total_feedback,
        "country_counts": country_counts,
        "user_type_counts": user_type_counts,
        "feedback_timeline": feedback_timeline,
        "last_feedback_date": last_feedback_date,
        "has_open_feedback": has_open_feedback,
        "feedback_ids": feedback_ids,
    }


def deduplicate(summaries):
    """Deduplicate initiatives sharing identical sorted feedback ID sets.

    Keeps the one with the most total_feedback; ties broken by higher ID.
    """
    seen = {}  # feedback_ids key -> index in summaries
    remove = set()

    for i, s in enumerate(summaries):
        key = ",".join(str(fid) for fid in s["feedback_ids"])
        if not key:
            continue
        prev = seen.get(key)
        if prev is not None:
            keep_prev = summaries[prev]["total_feedback"] > s["total_feedback"] or (
                summaries[prev]["total_feedback"] == s["total_feedback"]
                and summaries[prev]["id"] > s["id"]
            )
            if keep_prev:
                remove.add(i)
            else:
                remove.add(prev)
                seen[key] = i
        else:
            seen[key] = i

    return [s for i, s in enumerate(summaries) if i not in remove]


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute webapp initiative index."
    )
    parser.add_argument(
        "details_dir",
        help="Path to initiative_details/ directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/webapp/initiative_index.json",
        help="Output JSON file path (default: data/webapp/initiative_index.json)",
    )
    args = parser.parse_args()

    details_dir = args.details_dir
    if not os.path.isdir(details_dir):
        print(f"ERROR: {details_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in os.listdir(details_dir) if f.endswith(".json"))
    print(f"Processing {len(files)} initiative files...")

    summaries = []
    errors = 0
    for filename in files:
        filepath = os.path.join(details_dir, filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {filename}: {e}", file=sys.stderr)
            errors += 1
            continue

        summary = build_summary(data)
        if summary:
            summaries.append(summary)

    print(f"Built {len(summaries)} summaries ({errors} errors)")

    before = len(summaries)
    summaries = deduplicate(summaries)
    dupes = before - len(summaries)
    if dupes:
        print(f"Removed {dupes} duplicates ({len(summaries)} remaining)")

    # Sort by last_cached_at descending
    summaries.sort(
        key=lambda s: s["last_cached_at"] or "", reverse=True
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
