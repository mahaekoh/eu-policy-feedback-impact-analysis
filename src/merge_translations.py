"""Merge translated feedback attachment texts back into initiative detail JSON files.

Takes the translation output from translate_attachments.py and updates the
corresponding initiative JSON files in-place, replacing extracted_text with
the translation and preserving the original as extracted_text_before_translation.

Records where the translation is "NO TRANSLATION NEEDED" are skipped.

Supports multiple input modes:
  - Combined JSON file (output of translate_attachments.py)
  - Batch directory (the *_batches/ folder from translate_attachments.py)
  - Batch directory + --input-records (for old batch files missing publication_id)

Usage:
    # From combined output
    python3 src/merge_translations.py translated.json initiative_details/

    # From batch directory (new format with publication_id)
    python3 src/merge_translations.py translated_batches/ initiative_details/

    # From batch directory (old format without publication_id)
    python3 src/merge_translations.py translated_batches/ initiative_details/ --input-records translation_tasks.json

    # Dry run
    python3 src/merge_translations.py translated_batches/ initiative_details/ --dry-run
"""

import argparse
import json
import os
import sys

from text_utils import split_into_chunks

CHUNK_SIZE = 5000


def load_combined(report_path):
    """Load the combined translated JSON (one record per attachment).

    Returns list of dicts with at least: initiative_id, publication_id,
    feedback_id, attachment_id, extracted_text_translated.
    """
    with open(report_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded combined file: {len(records)} records")
    return records


def _list_batch_files(batch_dir):
    """List batch files in a directory, supporting both .json and extensionless names."""
    files = []
    for f in sorted(os.listdir(batch_dir)):
        path = os.path.join(batch_dir, f)
        if not os.path.isfile(path):
            continue
        if f.startswith("batch_"):
            files.append(f)
    return files


def load_from_batches(batch_dir, details_dir, chunk_size, input_records_path=None):
    """Load batch files and reassemble chunk translations into merge-ready records.

    Groups batch entries by attachment, reassembles chunks, and handles
    "NO TRANSLATION NEEDED" replacement using original text from initiative files.

    If batch entries lack publication_id (old format), it is looked up from
    --input-records or from the initiative detail files.

    Returns list of dicts with: initiative_id, publication_id, feedback_id,
    attachment_id, extracted_text_translated.
    """
    # Load all batch entries
    batch_files = _list_batch_files(batch_dir)
    if not batch_files:
        print(f"No batch files found in {batch_dir}")
        return []

    # Load input records if provided (for publication_id lookup)
    input_records_by_key = {}
    if input_records_path:
        with open(input_records_path, encoding="utf-8") as f:
            input_records = json.load(f)
        for i, rec in enumerate(input_records):
            key = (rec.get("initiative_id"), rec.get("feedback_id"), rec.get("attachment_id"))
            input_records_by_key[key] = rec
        print(f"Loaded input records: {len(input_records)} records")

    # Group entries by (initiative_id, feedback_id, attachment_id)
    by_attachment = {}  # key -> {"chunks": [...], "publication_id": ...}
    total_entries = 0
    for bf in batch_files:
        with open(os.path.join(batch_dir, bf), encoding="utf-8") as f:
            batch_results = json.load(f)
        for entry in batch_results:
            key = (entry["initiative_id"], entry["feedback_id"], entry["attachment_id"])
            if key not in by_attachment:
                by_attachment[key] = {
                    "chunks": [],
                    "publication_id": entry.get("publication_id"),
                }
            by_attachment[key]["chunks"].append(
                (entry["chunk_index"], entry["translation"])
            )
            total_entries += 1

    print(f"Loaded {len(batch_files)} batch files ({total_entries} chunk entries, "
          f"{len(by_attachment)} unique attachments)")

    # Cache loaded initiative files
    initiative_cache = {}

    def _load_initiative(init_id):
        if init_id not in initiative_cache:
            json_path = os.path.join(details_dir, f"{init_id}.json")
            if os.path.isfile(json_path):
                with open(json_path, encoding="utf-8") as f:
                    initiative_cache[init_id] = json.load(f)
            else:
                initiative_cache[init_id] = None
        return initiative_cache.get(init_id)

    def _find_pub_id_from_initiative(initiative, fb_id, att_id):
        """Search initiative structure for the publication containing this attachment."""
        for pub in initiative.get("publications", []):
            for fb in pub.get("feedback", []):
                if fb.get("id") != fb_id:
                    continue
                for att in fb.get("attachments", []):
                    if att.get("id") == att_id:
                        return pub.get("publication_id")
        return None

    # Reassemble per-attachment translations
    records = []
    missing_pub_id = 0
    for (init_id, fb_id, att_id), info in by_attachment.items():
        chunks = info["chunks"]
        pub_id = info["publication_id"]

        # Resolve publication_id if missing
        if pub_id is None:
            # Try input records first
            input_rec = input_records_by_key.get((init_id, fb_id, att_id))
            if input_rec:
                pub_id = input_rec.get("publication_id")

            # Fall back to searching initiative files
            if pub_id is None:
                initiative = _load_initiative(init_id)
                if initiative:
                    pub_id = _find_pub_id_from_initiative(initiative, fb_id, att_id)

            if pub_id is None:
                missing_pub_id += 1
                print(f"  WARNING: could not find publication_id for "
                      f"init={init_id} fb={fb_id} att={att_id}", file=sys.stderr)
                continue

        # Sort chunks by index
        chunks.sort(key=lambda x: x[0])
        n_chunks = max(ci for ci, _ in chunks) + 1

        # Check if any chunk needs original text substitution
        needs_original = any("NO TRANSLATION NEEDED" in t for _, t in chunks)
        original_chunks_map = {}

        if needs_original:
            initiative = _load_initiative(init_id)
            if initiative:
                original_text = _find_attachment_text(initiative, pub_id, fb_id, att_id)
                if original_text:
                    orig_chunks = split_into_chunks(original_text.strip(), chunk_size)
                    for ci, chunk in enumerate(orig_chunks):
                        original_chunks_map[ci] = chunk

        # Assemble final translation
        parts = []
        chunks_dict = {ci: t for ci, t in chunks}
        for ci in range(n_chunks):
            translation = chunks_dict.get(ci, "")
            if "NO TRANSLATION NEEDED" in translation:
                parts.append(original_chunks_map.get(ci, ""))
            else:
                parts.append(translation)

        combined = "\n\n".join(parts)

        records.append({
            "initiative_id": init_id,
            "publication_id": pub_id,
            "feedback_id": fb_id,
            "attachment_id": att_id,
            "extracted_text_translated": combined,
        })

    if missing_pub_id:
        print(f"WARNING: {missing_pub_id} attachments skipped (missing publication_id)")

    return records


def _find_attachment_text(initiative, pub_id, fb_id, att_id):
    """Look up extracted_text for an attachment in an initiative structure."""
    for pub in initiative.get("publications", []):
        if pub.get("publication_id") != pub_id:
            continue
        for fb in pub.get("feedback", []):
            if fb.get("id") != fb_id:
                continue
            for att in fb.get("attachments", []):
                if att.get("id") == att_id:
                    return att.get("extracted_text", "")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Merge translated texts into initiative detail JSON files."
    )
    parser.add_argument(
        "report",
        help="Path to combined translated JSON, or path to batch directory",
    )
    parser.add_argument(
        "details_dir", help="Directory of per-initiative JSON files"
    )
    parser.add_argument(
        "--input-records",
        help="Path to the original input JSON (for old batch files missing publication_id).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Chunk size used during translation (default: {CHUNK_SIZE}). "
             "Only relevant when using batch directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed changes without modifying files.",
    )
    args = parser.parse_args()

    # Determine input mode
    if os.path.isdir(args.report):
        records = load_from_batches(
            args.report, args.details_dir, args.chunk_size, args.input_records
        )
    else:
        records = load_combined(args.report)

    # Group records by initiative ID, skipping those without a usable translation
    by_initiative = {}
    skipped_no_translation = 0
    skipped_not_needed = 0
    for rec in records:
        translated = rec.get("extracted_text_translated", "")
        if not translated:
            skipped_no_translation += 1
            continue
        if "NO TRANSLATION NEEDED" in translated:
            skipped_not_needed += 1
            continue
        init_id = rec["initiative_id"]
        by_initiative.setdefault(init_id, []).append(rec)

    actionable = sum(len(v) for v in by_initiative.values())
    print(f"Records: {len(records)} total, {actionable} to merge, "
          f"{skipped_not_needed} already English, {skipped_no_translation} without translation")

    updated = 0
    skipped_lookup = 0
    modified_files = set()

    for init_id, recs in sorted(by_initiative.items()):
        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            skipped_lookup += len(recs)
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        # Index publications for fast lookup
        pubs_by_id = {}
        for pub in initiative.get("publications", []):
            pubs_by_id[pub.get("publication_id")] = pub

        changed = False
        for rec in recs:
            pub_id = rec["publication_id"]
            pub = pubs_by_id.get(pub_id)
            if not pub:
                print(f"  SKIP initiative {init_id}: pub {pub_id} not found", file=sys.stderr)
                skipped_lookup += 1
                continue

            fb_id = rec["feedback_id"]
            att_id = rec["attachment_id"]
            target = None
            for fb in pub.get("feedback", []):
                if fb.get("id") != fb_id:
                    continue
                for att in fb.get("attachments", []):
                    if att.get("id") == att_id:
                        target = att
                        break
                if target:
                    break

            if not target:
                print(f"  SKIP initiative {init_id}: fb {fb_id} att {att_id} not found", file=sys.stderr)
                skipped_lookup += 1
                continue

            old_text = target.get("extracted_text", "")
            translated = rec["extracted_text_translated"]

            if args.dry_run:
                old_snippet = old_text.replace("\n", " ").strip()[:120] if old_text else "(empty)"
                new_snippet = translated.replace("\n", " ").strip()[:120] if translated else "(empty)"
                print(f"\n[initiative {init_id}, pub {pub_id}, fb {fb_id}, att {att_id}]")
                print(f"  filename:   {rec.get('filename', '?')}")
                print(f"  language:   {rec.get('language', '?')}")
                print(f"  original  ({len(old_text)} chars): {old_snippet}")
                print(f"  translated ({len(translated)} chars): {new_snippet}")
            else:
                target["extracted_text_before_translation"] = old_text
                target["extracted_text"] = translated
                changed = True

            updated += 1

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nUpdated: {updated}, Skipped (lookup): {skipped_lookup}, Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
