"""Load the verified evaluation corpus into Supabase using original passage IDs."""

from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from .corpus import load_corpus

BATCH_SIZE = 250


def request(url, key, method, params, *, rows=None, prefer=None):
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        url + "?" + urllib.parse.urlencode(params),
        method=method,
        headers=headers,
        data=json.dumps(rows, ensure_ascii=False).encode() if rows is not None else None,
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
                return response.headers, json.loads(payload) if payload else None
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Passage request failed with HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, http.client.HTTPException):
            if attempt == 2:
                raise RuntimeError("Passage request failed after three connection attempts") from None
        time.sleep(2**attempt)


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")
    host = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not re.fullmatch(r"https://[a-z0-9]+\.supabase\.co", host) or not key:
        raise SystemExit("Set a hosted Supabase URL and SUPABASE_SERVICE_ROLE_KEY before importing.")
    url = f"{host}/rest/v1/passages"
    corpus = load_corpus()
    passages = dict(zip(corpus.ids, corpus.texts, strict=True))
    if any("\x00" in text for text in passages.values()):
        raise SystemExit("The corpus contains text that PostgreSQL cannot store unchanged.")
    rows = [{"original_passage_id": pid, "text": text} for pid, text in passages.items()]
    for offset in range(0, len(rows), BATCH_SIZE):
        batch = rows[offset:offset + BATCH_SIZE]
        request(url, key, "POST", {"on_conflict": "original_passage_id"}, rows=batch,
                prefer="resolution=ignore-duplicates,return=minimal")
        if offset % (10 * BATCH_SIZE) == 0 or offset + len(batch) == len(rows):
            print(f"Submitted {offset + len(batch):,}/{len(rows):,} passages", flush=True)
    headers, _ = request(url, key, "HEAD", {"select": "id", "limit": "1"}, prefer="count=exact")
    count = headers.get("Content-Range", "").split("/")[-1]
    if count != str(len(rows)):
        raise RuntimeError(f"Supabase passage count {count!r} differs from {len(rows):,}")
    sample_ids = corpus.ids[::max(1, len(rows) // 10)][:10]
    _, fetched = request(url, key, "GET", {
        "select": "original_passage_id,text",
        "original_passage_id": "in.(" + ",".join(json.dumps(pid) for pid in sample_ids) + ")",
    })
    if {row["original_passage_id"]: row["text"] for row in fetched} != {
        pid: passages[pid] for pid in sample_ids
    }:
        raise RuntimeError("Sampled passage IDs or text differ from the source corpus")
    print(f"Verified {count} rows and {len(sample_ids)} sampled passage texts.", flush=True)


if __name__ == "__main__":
    main()
