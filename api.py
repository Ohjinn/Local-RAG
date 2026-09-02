"""RAG 파이프라인을 HTTP 로 연다.

query.py 와 같은 일을 하되 CLI 대신 HTTP 로 부를 수 있게 한 것이다. 로직을
새로 쓰지 않고 같은 Chroma 컬렉션과 같은 프롬프트를 쓴다.

경로가 세 갈래인 이유는 필요한 것이 서로 다르기 때문이다.

    /search/text   Chroma 만    맥북 없이도 된다. 글자가 그대로 든 청크.
    /search        + 임베딩     맥북 필요. 뜻이 가까운 청크.
    /ask           + 생성       맥북 필요. 청크를 읽고 답을 쓴다.

맥북(임베딩·생성 서버)이 자고 있으면 뒤의 둘은 동작할 수 없다. 그때 스택
트레이스 대신 503 과 한국어 설명을 돌려준다 — 왜 안 되는지 모른 채 헤매는
일이 여러 번 있었다.

    uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

from chroma_store import chroma_kwargs, describe
from query import COLLECTION, EMBED_MODEL, LLM_MODEL, OLLAMA, PROMPT

Category = Literal["docs", "infra-code", "app-code", "history", "context"]

app = FastAPI(
    title="Local-RAG",
    description=(
        "홈랩 문서 RAG. 임베딩과 생성은 맥북 Ollama, 벡터 저장은 k3s 의 Chroma."
    ),
    version="1.0.0",
)

# 프로세스가 뜰 때 한 번만 만든다. 생성자는 네트워크를 건드리지 않으므로
# 맥북이 자고 있어도 기동은 된다 — 실제 호출 때 비로소 실패한다.
_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA),
    **chroma_kwargs(),
)


def _where(category: str | None, source: str | None) -> dict[str, Any] | None:
    """메타데이터 필터를 만든다. Chroma 는 조건이 둘 이상이면 $and 로 감싸야 한다."""
    conds = {}
    if category:
        conds["category"] = category
    if source:
        conds["source"] = source
    if len(conds) > 1:
        return {"$and": [{k: v} for k, v in conds.items()]}
    return conds or None


def _require_ollama() -> None:
    """맥북이 살아 있는지 먼저 확인한다.

    확인하지 않으면 langchain 이 재시도하다 한참 뒤에야 실패해서, 호출한 쪽은
    '느리다' 와 '안 된다' 를 구분하지 못한다.
    """
    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                f"임베딩·생성 서버({OLLAMA})에 닿지 않습니다. 맥북이 꺼져 있거나 "
                f"자는 중입니다. /search/text 는 맥북 없이도 동작합니다."
            ),
        )


def _fmt(doc, score=None) -> dict[str, Any]:
    m = doc.metadata or {}
    out = {
        "breadcrumb": m.get("breadcrumb") or m.get("path"),
        "path": m.get("path"),
        "source": m.get("source"),
        "category": m.get("category"),
        "content": doc.page_content,
    }
    if score is not None:
        # l2 거리다. 코사인 유사도가 아니므로 낮을수록 가깝다.
        out["distance"] = round(float(score), 4)
    return out


# ─────────────────────────────── 상태 ───────────────────────────────


@app.get("/health", summary="의존 서비스 상태")
def health() -> dict[str, Any]:
    """Chroma 와 맥북 Ollama 를 각각 확인한다.

    한쪽만 죽어도 되는 기능이 갈리므로 뭉뚱그리지 않고 따로 보고한다.
    """
    out: dict[str, Any] = {"store": describe()}

    try:
        out["chunks"] = _store._collection.count()
        out["chroma"] = "ok"
    except Exception as e:
        out["chroma"] = f"error: {type(e).__name__}"

    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        out["ollama"] = "ok"
        out["models"] = {
            "embed": EMBED_MODEL,
            "llm": LLM_MODEL,
            "embed_loaded": any(n.startswith(EMBED_MODEL) for n in names),
            "llm_loaded": LLM_MODEL in names,
        }
    except Exception:
        out["ollama"] = "unreachable"
        out["note"] = "맥북이 자는 중입니다. /search/text 만 동작합니다."

    return out


# ────────────────────────── 검색 (LLM 없음) ──────────────────────────


@app.get("/search/text", summary="전문 검색 — 맥북 불필요")
def search_text(
    q: str = Query(..., description="이 글자가 그대로 든 청크를 찾는다"),
    n: int = Query(5, ge=1, le=50),
    category: Category | None = None,
    source: str | None = Query(None, description="레포 이름 (예: wizparking)"),
) -> dict[str, Any]:
    """글자로 찾는다. 임베딩을 하지 않으므로 맥북이 없어도 된다.

    "hermes" 로는 찾지만 "헤르메스" 로는 못 찾는다. 정확한 식별자
    (ImagePullBackOff, 커밋 해시, 변수명)를 찾을 때 쓴다. 순위는 관련도가
    아니라 저장 순서다 — 거리 계산을 하지 않기 때문이다.
    """
    t0 = time.perf_counter()
    res = _store.get(
        where=_where(category, source),
        where_document={"$contains": q},
        limit=n,
        include=["metadatas", "documents"],
    )
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return {
        "mode": "fulltext",
        "query": q,
        "count": len(docs),
        "took_ms": round((time.perf_counter() - t0) * 1000),
        "results": [
            {
                "breadcrumb": (m or {}).get("breadcrumb") or (m or {}).get("path"),
                "path": (m or {}).get("path"),
                "source": (m or {}).get("source"),
                "category": (m or {}).get("category"),
                "content": d,
            }
            for d, m in zip(docs, metas)
        ],
    }


@app.get("/search", summary="의미 검색 — 청크만, 답 생성 없음")
def search(
    q: str = Query(..., description="질문. 글자가 안 겹쳐도 뜻이 가까우면 찾는다"),
    k: int = Query(6, ge=1, le=50),
    category: Category | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """벡터 유사도로 찾는다. 질문을 맥북에서 bge-m3 로 임베딩한 뒤 넘긴다.

    LLM 을 거치지 않으므로 /ask 보다 훨씬 빠르다. 검색이 제대로 되는지만
    보고 싶을 때 답 생성을 기다릴 이유가 없다.
    """
    _require_ollama()
    t0 = time.perf_counter()
    hits = _store.similarity_search_with_score(q, k=k, filter=_where(category, source))
    return {
        "mode": "vector",
        "query": q,
        "count": len(hits),
        "took_ms": round((time.perf_counter() - t0) * 1000),
        "results": [_fmt(d, s) for d, s in hits],
    }


# ──────────────────────────── 질의 (LLM) ────────────────────────────


class AskRequest(BaseModel):
    question: str
    k: int = Field(6, ge=1, le=30)
    category: Category | None = None
    source: str | None = None


def _retrieve(question: str, k: int, category, source):
    _require_ollama()
    t0 = time.perf_counter()
    hits = _store.similarity_search_with_score(
        question, k=k, filter=_where(category, source)
    )
    return hits, time.perf_counter() - t0


def _llm() -> ChatOllama:
    # keep_alive 30분: 14b 는 9GB 라 메모리에서 내려가면 다시 올리는 데만
    # 14초쯤 걸린다. 띄엄띄엄 물어보면 매번 그 값을 문다.
    return ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA, temperature=0, keep_alive="30m"
    )


@app.post("/ask", summary="RAG 질의 — 검색 + 답 생성")
def ask(req: AskRequest) -> dict[str, Any]:
    """검색한 청크를 근거로 LLM 이 답을 쓴다. query.py 와 같은 프롬프트다.

    맥북 14b 로 30초 안팎 걸린다. 근거만 필요하면 /search 를 쓰는 편이 빠르다.
    """
    hits, t_search = _retrieve(req.question, req.k, req.category, req.source)
    if not hits:
        return {
            "question": req.question,
            "answer": None,
            "sources": [],
            "note": "검색 결과가 없어 답을 만들지 않았습니다.",
        }

    context = "\n\n---\n\n".join(d.page_content for d, _ in hits)
    t0 = time.perf_counter()
    answer = _llm().invoke(PROMPT.format(context=context, question=req.question))
    t_gen = time.perf_counter() - t0

    return {
        "question": req.question,
        "answer": answer.content,
        "sources": [_fmt(d, s) for d, s in hits],
        "timing": {
            "search_ms": round(t_search * 1000),
            "generate_ms": round(t_gen * 1000),
            "total_ms": round((t_search + t_gen) * 1000),
        },
        "context_chars": len(context),
    }


@app.get("/ask", summary="RAG 질의 (GET) — 브라우저·curl 로 바로")
def ask_get(
    q: str = Query(..., description="질문"),
    k: int = Query(6, ge=1, le=30),
    category: Category | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """POST /ask 와 같다. 주소창이나 curl 한 줄로 부를 수 있게 열어 둔다."""
    return ask(AskRequest(question=q, k=k, category=category, source=source))


# ───────────────────── OpenAI 호환 (Open WebUI 등) ─────────────────────
#
# 이 경로가 있으면 litellm 에 모델 하나로 등록할 수 있고, Open WebUI 나 HA
# Assist 에서 "홈랩을 아는 모델" 로 그냥 골라 쓸 수 있다. 클라이언트를 새로
# 만들 필요가 없다는 것이 이 형식을 맞추는 유일한 이유다.

MODEL_ID = "homelab-rag"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    # 나머지 OpenAI 파라미터(temperature 등)는 받되 무시한다. 거절하면
    # 클라이언트가 붙지 않는다 — litellm 폴백이 파라미터 거절로 통째로
    # 실패했던 것과 같은 함정이다.
    model_config = {"extra": "allow"}


@app.get("/v1/models", summary="OpenAI 호환 — 모델 목록")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "homelab"}],
    }


@app.post("/v1/chat/completions", summary="OpenAI 호환 — RAG 를 모델처럼")
def chat_completions(req: ChatRequest):
    """마지막 user 메시지를 질문으로 삼아 RAG 를 돌린다.

    대화 이력은 검색에 쓰지 않는다. 앞선 턴까지 임베딩하면 질문이 흐려져
    엉뚱한 청크가 끌려온다.
    """
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "user 메시지가 없습니다.")
    question = user_msgs[-1]

    hits, _ = _retrieve(question, 6, None, None)
    context = "\n\n---\n\n".join(d.page_content for d, _ in hits)
    prompt = PROMPT.format(context=context, question=question)

    if not req.stream:
        answer = _llm().invoke(prompt)
        return {
            "id": "chatcmpl-homelab",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer.content},
                    "finish_reason": "stop",
                }
            ],
        }

    def sse():
        for part in _llm().stream(prompt):
            if not part.content:
                continue
            chunk = {
                "id": "chatcmpl-homelab",
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": part.content}}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        done = {
            "id": "chatcmpl-homelab",
            "object": "chat.completion.chunk",
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
