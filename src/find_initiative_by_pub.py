"""Find the initiative that contains a given publication ID."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_initiative_by_pub.py <publication_id>")
        sys.exit(1)

    pub_id = int(sys.argv[1])
    jsonl_path = Path(__file__).parent.parent / "eu_initiative_details.jsonl"

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for pub in obj.get("publications", []):
                if pub.get("publication_id") == pub_id:
                    init_id = obj["id"]
                    title = obj.get("short_title", "")
                    pub_type = pub["type"]
                    n_docs = len(pub.get("documents", []))
                    n_fb = len(pub.get("feedback", []))
                    print(f"Initiative {init_id}: {title}")
                    print(f"  https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/{init_id}")
                    print(f"  Publication {pub_id}: type={pub_type}, {n_docs} docs, {n_fb} feedback")
                    return

    print(f"Publication {pub_id} not found")


if __name__ == "__main__":
    main()
