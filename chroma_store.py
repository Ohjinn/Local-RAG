"""Chroma 저장소를 로컬 디렉토리 또는 원격 서버 중 하나로 연결한다.

`CHROMA_URL` 이 있으면 원격(HttpClient), 없으면 기존처럼 로컬 디렉토리를 쓴다.
로컬을 남겨 두는 이유는 맥북이 집에 없을 때 서버가 있어도 임베딩이 안 되고,
반대로 서버가 내려가 있어도 손으로 돌려볼 수 있어야 하기 때문이다.

    export CHROMA_URL=http://chroma.newhojin.com
    export CHROMA_TOKEN=$(cat ~/.chroma_token)

클러스터 안에서 도는 것(인덱싱 CronJob)은 인그레스를 탈 이유가 없으므로
http://chroma.chroma-system.svc:8000 을 쓴다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
LOCAL_DIR = ROOT / "chroma_db"
COLLECTION = "homelab"


def chroma_kwargs():
    """langchain_chroma.Chroma() 에 넘길 인자를 만든다.

    반환값에 collection_name 은 넣지 않는다 — 호출부가 이미 지정하고 있고,
    중복으로 넘기면 TypeError 가 난다.
    """
    url = os.environ.get("CHROMA_URL", "").strip()
    if not url:
        return {"persist_directory": str(LOCAL_DIR)}

    import chromadb
    from chromadb.config import Settings
    from urllib.parse import urlparse

    u = urlparse(url)
    token = os.environ.get("CHROMA_TOKEN", "").strip()
    if not token:
        p = Path.home() / ".chroma_token"
        if p.exists():
            token = p.read_text().strip()

    settings = Settings(anonymized_telemetry=False)
    headers = {}
    if token:
        # 서버의 CHROMA_AUTH_TOKEN_TRANSPORT_HEADER 와 같아야 한다.
        headers["X-Chroma-Token"] = token

    client = chromadb.HttpClient(
        host=u.hostname,
        port=u.port or (443 if u.scheme == "https" else 80),
        ssl=(u.scheme == "https"),
        headers=headers,
        settings=settings,
    )
    return {"client": client}


def describe():
    url = os.environ.get("CHROMA_URL", "").strip()
    return url if url else "로컬 %s" % LOCAL_DIR
