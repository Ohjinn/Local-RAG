#!/usr/bin/env python3
"""로컬 chroma_db/ 의 컬렉션을 원격 Chroma 서버로 그대로 옮긴다.

임베딩을 다시 계산하지 않는다. 벡터가 이미 있으므로 문서·메타데이터와 함께
복사만 하면 된다. 맥북(임베딩 서버)이 집에 없어도 옮길 수 있는 이유다.

    export CHROMA_URL=http://chroma.newhojin.com
    export CHROMA_TOKEN=$(cat ~/.chroma_token)
    uv run migrate_to_server.py [--batch 200] [--force]
"""
import argparse
import os
import sys
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings

from chroma_store import COLLECTION, LOCAL_DIR


def remote_client():
    url = os.environ.get("CHROMA_URL", "").strip()
    if not url:
        sys.exit("CHROMA_URL 이 필요하다 (예: http://chroma.newhojin.com)")
    u = urlparse(url)
    token = os.environ.get("CHROMA_TOKEN", "").strip()
    if not token:
        from pathlib import Path
        p = Path.home() / ".chroma_token"
        if p.exists():
            token = p.read_text().strip()
    headers = {"X-Chroma-Token": token} if token else {}
    return chromadb.HttpClient(
        host=u.hostname, port=u.port or (443 if u.scheme == "https" else 80),
        ssl=(u.scheme == "https"), headers=headers,
        settings=Settings(anonymized_telemetry=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--force", action="store_true",
                    help="원격에 이미 데이터가 있어도 지우고 다시 넣는다")
    args = ap.parse_args()

    src = chromadb.PersistentClient(path=str(LOCAL_DIR),
                                    settings=Settings(anonymized_telemetry=False))
    s = src.get_collection(COLLECTION)
    total = s.count()
    print("로컬  %s → %d 개" % (LOCAL_DIR, total))
    if total == 0:
        sys.exit("로컬 컬렉션이 비어 있다 — 옮길 게 없다")

    dst = remote_client()
    print("원격  %s" % os.environ["CHROMA_URL"])

    existing = None
    try:
        existing = dst.get_collection(COLLECTION)
    except Exception:
        pass
    if existing is not None:
        n = existing.count()
        if n and not args.force:
            sys.exit("원격에 이미 %d 개가 있다. 덮어쓰려면 --force" % n)
        if n:
            dst.delete_collection(COLLECTION)
            existing = None
            print("  기존 %d 개를 지웠다" % n)
    if existing is None:
        # 임베딩 함수를 붙이지 않는다. 이미 계산된 벡터를 직접 넣기 때문에
        # 서버가 임베딩을 시도하면 안 된다(맥북이 없으면 실패한다).
        d = dst.create_collection(COLLECTION, metadata=s.metadata or None)
    else:
        d = existing

    moved = 0
    while moved < total:
        got = s.get(limit=args.batch, offset=moved,
                    include=["embeddings", "documents", "metadatas"])
        ids = got["ids"]
        if not ids:
            break
        d.add(ids=ids,
              embeddings=got["embeddings"],
              documents=got["documents"],
              metadatas=got["metadatas"])
        moved += len(ids)
        print("  %d / %d" % (moved, total))

    print()
    print("옮김: %d 개" % moved)
    print("원격 확인: %d 개" % dst.get_collection(COLLECTION).count())


if __name__ == "__main__":
    main()
