"""인덱스에 질문한다.

    python query.py "traefik access log 는 왜 켰나"
    python query.py -c history "hermes 가 왜 안 뜨나"      # 커밋 이력에서만
    python query.py -k 10 "NSG 아웃바운드 규칙 정책"        # 근거를 더 많이

카테고리 필터가 있는 이유는 질문 종류가 갈리기 때문이다. "왜 이렇게
했지" 는 history 와 docs 에 답이 있고, "그 값이 뭐지" 는 infra-code 에
있다. 섞어서 검색하면 서로를 밀어낸다.
"""

from __future__ import annotations

import argparse
import os
import time

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from chroma_store import chroma_kwargs, describe

COLLECTION = "homelab"
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://192.168.0.44:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED", "bge-m3")

# qwen2.5 를 쓰는 이유는 두 가지다. 한국어 출력이 llama3.1 보다 자연스럽고,
# "주어진 문서에만 근거하라" 는 제약을 더 잘 지킨다. RAG 에서는 지어내지
# 않는 쪽이 유창한 쪽보다 중요하다.
LLM_MODEL = os.environ.get("RAG_LLM", "qwen2.5:14b")

PROMPT = """너는 hojin 의 홈랩과 인프라 코드를 파악하고 있는 조수다.

아래 <문서> 에 있는 내용만 근거로 답한다. 문서에 없는 것은 지어내지 말고
"문서에 없다" 고 분명히 말해라. 추측을 사실처럼 쓰지 마라.

답할 때는 어느 출처에서 나온 내용인지 함께 밝혀라. 각 문서의 첫 줄이
출처 경로다.

<문서>
{context}
</문서>

질문: {question}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=6, help="검색할 청크 수 (기본 6)")
    ap.add_argument(
        "-c",
        "--category",
        choices=["docs", "infra-code", "app-code", "history", "context"],
        help="이 카테고리에서만 검색",
    )
    ap.add_argument("-s", "--source", help="이 출처에서만 검색 (예: wizparking)")
    ap.add_argument("--show", action="store_true", help="검색된 청크를 그대로 출력")
    args = ap.parse_args()

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA),
        **chroma_kwargs(),
    )

    where = {}
    if args.category:
        where["category"] = args.category
    if args.source:
        where["source"] = args.source
    # Chroma 는 조건이 둘 이상이면 $and 로 감싸야 한다.
    if len(where) > 1:
        where = {"$and": [{k: v} for k, v in where.items()]}

    # 검색과 생성을 따로 잰다. 느릴 때 임베딩 왕복이 문제인지 모델이
    # 문제인지 갈라 봐야 손댈 곳이 정해진다.
    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(
        args.question, k=args.k, filter=where or None
    )
    t_search = time.perf_counter() - t0

    if not hits:
        print(f"검색 결과 없음. (검색 {t_search:.2f}s)")
        return

    print(f"근거 {len(hits)} 개 (검색 {t_search:.2f}s):")
    for doc, score in hits:
        print(f"  [{score:.3f}] {doc.metadata['breadcrumb']}")
    print()

    if args.show:
        for doc, _ in hits:
            print("─" * 70)
            print(doc.page_content)
        print("─" * 70)
        print()

    # Ollama 는 5 분 놀면 모델을 메모리에서 내린다. 14b 는 9GB 라 다시
    # 올리는 데만 14 초쯤 걸려서, 띄엄띄엄 물어보면 매번 그 값을 문다.
    # 맥북 18GB 에 물려 둘 만한 크기라 30 분 붙잡아 둔다.
    llm = ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA, temperature=0, keep_alive="30m"
    )
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in hits)

    print("답변:")
    t1 = time.perf_counter()
    t_first = None
    chars = 0
    for part in llm.stream(PROMPT.format(context=context, question=args.question)):
        if t_first is None and part.content:
            # 첫 토큰까지 걸린 시간. 사용자가 "멈춰 있다" 고 느끼는 구간이
            # 전체 시간이 아니라 이 구간이다.
            t_first = time.perf_counter() - t1
        chars += len(part.content)
        print(part.content, end="", flush=True)
    t_gen = time.perf_counter() - t1

    print(
        f"\n\n[검색 {t_search:.2f}s · 첫 토큰 {t_first or 0:.2f}s · "
        f"생성 {t_gen:.2f}s · 합계 {t_search + t_gen:.2f}s · "
        f"{chars / t_gen:.0f}자/s · 컨텍스트 {len(context):,}자]"
    )


if __name__ == "__main__":
    main()
