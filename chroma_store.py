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

    # Chroma 1.0 부터 내장 인증이 없어졌다(Rust 재작성). 그래서 집에서는
    # traefik basicAuth 미들웨어가 앞단에서 막고, 자격증명은 URL 에 실어
    # 보낸다: http://user:pass@chroma.newhojin.com
    #
    # 클러스터 안에서는 Service 로 직접 붙어 인그레스를 안 타므로 자격증명이
    # 없고, 단독 배포에서는 localhost 바인딩이라 노출 자체가 없다. 셋 다
    # 같은 코드로 처리된다 — URL 에 user:pass 가 있으면 보내고 없으면 만다.
    if u.username:
        import base64
        raw = "%s:%s" % (u.username, u.password or "")
        headers["Authorization"] = "Basic " + base64.b64encode(raw.encode()).decode()

    if token:
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
    """로그에 찍을 저장소 설명.

    URL 에 든 비밀번호는 반드시 가린다. 가리지 않은 채로 출력했다가 실제로
    자격증명을 노출시켜 교체해야 했다.
    """
    url = os.environ.get("CHROMA_URL", "").strip()
    if not url:
        return "로컬 %s" % LOCAL_DIR
    import re
    return re.sub(r"(//[^:/@]+:)[^@]*@", r"\1***@", url)
