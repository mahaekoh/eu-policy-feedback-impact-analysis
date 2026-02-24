"""Print a specific chunk for a feedback attachment.

Usage:
    python3 src/print_chunk.py "init=12096 fb=503089 att=6276475 chunk=5/15" initiative_details/
    python3 src/print_chunk.py "init=12096 fb=503089 att=6276475 chunk=5/15" initiative_details/ --chunk-size 8000
"""

import argparse
import json
import re
import sys
from pathlib import Path

from text_utils import split_into_chunks

CHUNK_SIZE = 5000


def main():
    parser = argparse.ArgumentParser(
        description="Print a specific chunk for a feedback attachment."
    )
    parser.add_argument(
        "spec",
        help='Attachment spec, e.g. "init=12096 fb=503089 att=6276475 chunk=5/15"',
    )
    parser.add_argument(
        "details_dir", help="Directory of per-initiative JSON files"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Max chars per chunk (default: {CHUNK_SIZE}).",
    )
    args = parser.parse_args()

    # Parse spec
    m_init = re.search(r'init=(\d+)', args.spec)
    m_fb = re.search(r'fb=(\d+)', args.spec)
    m_att = re.search(r'att=(\d+)', args.spec)
    m_chunk = re.search(r'chunk=(\d+)/(\d+)', args.spec)

    if not m_init or not m_fb or not m_att:
        print("Error: spec must contain init=, fb=, and att=", file=sys.stderr)
        sys.exit(1)

    init_id = int(m_init.group(1))
    fb_id = int(m_fb.group(1))
    att_id = int(m_att.group(1))
    chunk_idx = int(m_chunk.group(1)) if m_chunk else 0

    # Load initiative
    json_path = Path(args.details_dir) / f"{init_id}.json"
    if not json_path.is_file():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        initiative = json.load(f)

    # Find the attachment
    target = None
    for pub in initiative.get("publications", []):
        for fb in pub.get("feedback", []):
            if fb.get("id") != fb_id:
                continue
            for att in fb.get("attachments", []):
                if att.get("id") == att_id:
                    target = att
                    break
            if target:
                break
        if target:
            break

    if not target:
        print(f"Error: attachment not found (init={init_id}, fb={fb_id}, att={att_id})", file=sys.stderr)
        sys.exit(1)

    text = target.get("extracted_text", "")
    if not text:
        print("Attachment has no extracted_text")
        sys.exit(0)

    chunks = split_into_chunks(text.strip(), args.chunk_size)

    print(f"Attachment: init={init_id} fb={fb_id} att={att_id}")
    print(f"Filename: {target.get('filename', '?')}")
    print(f"Total text: {len(text)} chars")
    print(f"Chunks: {len(chunks)} (chunk size: {args.chunk_size})")
    print()

    if chunk_idx >= len(chunks):
        print(f"Error: chunk {chunk_idx} out of range (0-{len(chunks)-1})", file=sys.stderr)
        sys.exit(1)

    print(f"--- Chunk {chunk_idx}/{len(chunks)} ({len(chunks[chunk_idx])} chars) ---")
    print(chunks[chunk_idx])


if __name__ == "__main__":
    main()
