"""Pre-compute the webapp initiative index and stripped initiative details.

Reads all initiative detail JSONs from data/scrape/initiative_details/,
extracts metadata, computes feedback statistics, deduplicates initiatives
sharing identical feedback IDs, and writes a single JSON index file for
the webapp to consume.

Also writes stripped copies of each initiative detail JSON to
data/webapp/initiative_details/, removing extracted_text fields from
feedback attachments to keep file sizes manageable for the webapp.

Usage:
    python3 src/build_webapp_index.py data/scrape/initiative_details/ -o data/webapp/initiative_index.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
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
    feedback_months = []
    feedback_country_months = []

    feedback_items = []

    short_title = data.get("short_title", "")

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
                    month = dt.strftime("%Y-%m")
                    feedback_months.append(month)
                    if country:
                        feedback_country_months.append((country, month))

            if country:
                fb_text = fb.get("feedback_text") or ""
                attachments = [
                    {
                        "filename": att.get("filename", ""),
                        "download_url": att.get("download_url", ""),
                    }
                    for att in fb.get("attachments", [])
                    if att.get("download_url")
                ]
                feedback_items.append({
                    "country": country,
                    "date": date or "",
                    "user_type": user_type or "",
                    "organization": fb.get("organization"),
                    "first_name": fb.get("first_name"),
                    "surname": fb.get("surname"),
                    "initiative_id": init_id,
                    "initiative_title": short_title,
                    "feedback_text": fb_text[:150] if fb_text else None,
                    "attachments": attachments,
                    "url": fb.get("url") or None,
                })

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
        "_feedback_months": feedback_months,
        "_feedback_country_months": feedback_country_months,
        "_feedback_items": feedback_items,
    }


ATTACHMENT_TEXT_FIELDS = (
    "extracted_text",
    "extracted_text_without_ocr",
    "extracted_text_before_translation",
)


def strip_large_text_fields(data):
    """Remove bulky text fields not needed by the webapp."""
    for pub in data.get("publications", []):
        for fb in pub.get("feedback", []):
            for att in fb.get("attachments", []):
                for field in ATTACHMENT_TEXT_FIELDS:
                    att.pop(field, None)
    for fb in data.get("middle_feedback", []):
        for att in fb.get("attachments", []):
            for field in ATTACHMENT_TEXT_FIELDS:
                att.pop(field, None)


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


def _sorted_counts(counter, limit=None):
    """Return a counter dict as a list of [key, count] pairs sorted desc."""
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if limit:
        items = items[:limit]
    return [[k, v] for k, v in items]


def _sorted_nested(nested, outer_limit=None, inner_limit=10):
    """Sort a nested {key: counter} dict by outer total desc, inner desc."""
    totals = {k: sum(v.values()) for k, v in nested.items()}
    keys = sorted(totals, key=totals.get, reverse=True)
    if outer_limit:
        keys = keys[:outer_limit]
    return {k: _sorted_counts(nested[k], inner_limit) for k in keys}


def build_global_stats(summaries):
    """Aggregate cross-initiative statistics from deduplicated summaries."""
    total_initiatives = len(summaries)
    total_feedback = 0
    by_country = defaultdict(int)
    by_topic = defaultdict(int)
    initiatives_by_topic = defaultdict(int)
    by_user_type = defaultdict(int)
    by_department = defaultdict(int)
    by_stage = defaultdict(int)
    topic_by_country = defaultdict(lambda: defaultdict(int))
    country_by_topic = defaultdict(lambda: defaultdict(int))
    by_month = defaultdict(int)
    # topic × month and country × month for time-series breakdowns
    topic_month = defaultdict(lambda: defaultdict(int))
    country_month = defaultdict(lambda: defaultdict(int))

    for s in summaries:
        total_feedback += s["total_feedback"]
        topics = s.get("topics", [])
        dept = s.get("department", "")
        stage = s.get("stage", "")

        if dept:
            by_department[dept] += s["total_feedback"]
        if stage:
            by_stage[stage] += 1  # count initiatives, not feedback

        for country, count in s["country_counts"].items():
            by_country[country] += count
            for topic in topics:
                topic_by_country[country][topic] += count
                country_by_topic[topic][country] += count

        for ut, count in s["user_type_counts"].items():
            by_user_type[ut] += count

        for topic in topics:
            by_topic[topic] += s["total_feedback"]
            initiatives_by_topic[topic] += 1

        for month in s.get("_feedback_months", []):
            by_month[month] += 1
            for topic in topics:
                topic_month[topic][month] += 1

        for country, month in s.get("_feedback_country_months", []):
            country_month[country][month] += 1

    # Build time-series with shared month axis, top 10 series each
    all_months = sorted(by_month.keys())
    top_topics = [t for t, _ in _sorted_counts(by_topic, limit=10)]
    top_countries = [c for c, _ in _sorted_counts(by_country, limit=15)]

    def _build_time_series(keys, month_data):
        return {
            "months": all_months,
            "series": {
                k: [month_data[k].get(m, 0) for m in all_months]
                for k in keys
            },
        }

    return {
        "total_initiatives": total_initiatives,
        "total_feedback": total_feedback,
        "by_country": _sorted_counts(by_country),
        "by_topic": _sorted_counts(by_topic),
        "initiatives_by_topic": _sorted_counts(initiatives_by_topic),
        "by_user_type": _sorted_counts(by_user_type),
        "by_department": _sorted_counts(by_department),
        "by_stage": _sorted_counts(by_stage),
        "top_topics_by_country": _sorted_nested(topic_by_country),
        "top_countries_by_topic": _sorted_nested(country_by_topic),
        "feedback_by_month": sorted(by_month.items()),
        "feedback_by_month_by_topic": _build_time_series(top_topics, topic_month),
        "feedback_by_month_by_country": _build_time_series(top_countries, country_month),
    }


def build_country_stats(summaries, all_months):
    """Build per-country statistics from deduplicated summaries."""
    # Accumulate per-country data
    country_data = defaultdict(lambda: {
        "total_feedback": 0,
        "topic_counts": defaultdict(int),
        "user_type_counts": defaultdict(int),
        "initiative_counts": defaultdict(int),
        "initiative_titles": {},
        "feedback_items": [],
        "topic_month": defaultdict(lambda: defaultdict(int)),
    })

    for s in summaries:
        topics = s.get("topics", [])
        for item in s.get("_feedback_items", []):
            country = item["country"]
            cd = country_data[country]
            cd["total_feedback"] += 1

            for topic in topics:
                cd["topic_counts"][topic] += 1

            ut = item.get("user_type")
            if ut:
                cd["user_type_counts"][ut] += 1

            init_id = item["initiative_id"]
            cd["initiative_counts"][init_id] += 1
            cd["initiative_titles"][init_id] = item["initiative_title"]

            cd["feedback_items"].append(item)

            date_str = item.get("date", "")
            if date_str:
                dt = parse_date(date_str)
                if dt:
                    month = dt.strftime("%Y-%m")
                    for topic in topics:
                        cd["topic_month"][topic][month] += 1

    result = {}
    for country, cd in country_data.items():
        # Top 20 topics
        by_topic = _sorted_counts(cd["topic_counts"], limit=20)

        # All user types
        by_user_type = _sorted_counts(cd["user_type_counts"])

        # Top 20 initiatives
        init_sorted = sorted(
            cd["initiative_counts"].items(), key=lambda x: x[1], reverse=True
        )[:20]
        top_initiatives = [
            {
                "id": init_id,
                "short_title": cd["initiative_titles"].get(init_id, ""),
                "count": count,
            }
            for init_id, count in init_sorted
        ]

        # 20 most recent feedback
        items_sorted = sorted(
            cd["feedback_items"], key=lambda x: x.get("date", ""), reverse=True
        )[:20]
        recent_feedback = [
            {
                "date": it["date"],
                "user_type": it["user_type"],
                "organization": it.get("organization"),
                "first_name": it.get("first_name"),
                "surname": it.get("surname"),
                "initiative_id": it["initiative_id"],
                "initiative_title": it["initiative_title"],
                "feedback_text": it.get("feedback_text"),
                "url": it.get("url"),
                "attachments": it.get("attachments", []),
            }
            for it in items_sorted
        ]

        # Topic timeline — top 5 topics over time
        top_5_topics = [t for t, _ in _sorted_counts(cd["topic_counts"], limit=5)]
        topic_timeline = {
            "months": all_months,
            "series": {
                t: [cd["topic_month"][t].get(m, 0) for m in all_months]
                for t in top_5_topics
            },
        }

        result[country] = {
            "total_feedback": cd["total_feedback"],
            "by_topic": by_topic,
            "by_user_type": by_user_type,
            "top_initiatives": top_initiatives,
            "recent_feedback": recent_feedback,
            "topic_timeline": topic_timeline,
        }

    return result


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

    # Stripped details directory sits next to the index file
    output_dir = os.path.dirname(args.output) or "."
    stripped_dir = os.path.join(output_dir, "initiative_details")
    os.makedirs(stripped_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(details_dir) if f.endswith(".json"))
    print(f"Processing {len(files)} initiative files...")

    summaries = []
    errors = 0
    n_files = len(files)
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"  Loading {i}/{n_files}...")
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

        # Write stripped copy (mutates data in-place, after summary is built)
        strip_large_text_fields(data)
        stripped_path = os.path.join(stripped_dir, filename)
        with open(stripped_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    print(f"Built {len(summaries)} summaries ({errors} errors)")

    before = len(summaries)
    summaries = deduplicate(summaries)
    dupes = before - len(summaries)
    if dupes:
        print(f"Removed {dupes} duplicates ({len(summaries)} remaining)")

    # Build global stats before stripping temporary fields
    print("Computing global statistics...")
    global_stats = build_global_stats(summaries)

    # Build country stats (needs _feedback_items + all_months from global stats)
    print("Computing per-country statistics...")
    all_months = [m for m, _ in global_stats["feedback_by_month"]]
    country_stats = build_country_stats(summaries, all_months)

    # Strip temporary fields and sort
    for s in summaries:
        s.pop("_feedback_months", None)
        s.pop("_feedback_country_months", None)
        s.pop("_feedback_items", None)
    summaries.sort(
        key=lambda s: s["last_cached_at"] or "", reverse=True
    )

    print(f"Writing {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")

    # Write global stats alongside the index
    print("Writing global_stats.json...")
    stats_path = os.path.join(output_dir, "global_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False)

    stats_mb = os.path.getsize(stats_path) / (1024 * 1024)
    print(f"Wrote {stats_path} ({stats_mb:.1f} MB)")

    # Write per-country stats
    print("Writing country_stats.json...")
    country_stats_path = os.path.join(output_dir, "country_stats.json")
    with open(country_stats_path, "w", encoding="utf-8") as f:
        json.dump(country_stats, f, ensure_ascii=False)

    country_stats_mb = os.path.getsize(country_stats_path) / (1024 * 1024)
    print(f"Wrote {country_stats_path} ({country_stats_mb:.1f} MB, {len(country_stats)} countries)")


if __name__ == "__main__":
    main()
