"""data/ 를 카테고리별로 잘라 Chroma 에 넣는다.

collect_docs.py 가 만든 manifest.json 을 읽는다. 디렉토리를 glob 하지
않는 이유는, 무엇이 어떤 카테고리인지가 manifest 에만 있기 때문이다.

카테고리마다 자르는 방식이 다르다:

  docs / context / history   마크다운 헤더 경계. history 는 커밋 하나가
                             `##` 섹션 하나라 자연히 커밋 단위로 떨어진다.
  infra-code                 resource / module / YAML 문서 경계
  app-code                   함수와 클래스 경계

청크 앞에는 breadcrumb 를 붙인다. 이게 검색 품질에 제일 크게 작용한다.
`### NSG 규칙` 하위 청크의 본문이 "아웃바운드는 명시 요청 없으면 추가하지
않는다" 한 줄이면, 그 벡터 안에는 어느 프로젝트의 무슨 NSG 인지가 전혀
없다. 조상 헤더와 파일 경로를 본문 앞에 붙여 두면 그 단어들이 벡터에
들어가고, 깊은 청크일수록 오히려 검색에 강해진다.

헤더 깊이(depth)는 메타데이터로 남기되 점수에는 쓰지 않는다. 깊다는 건
덜 중요하다는 뜻이 아니라 더 구체적이라는 뜻이고, 질문도 대개 구체적이라
깊은 청크가 정답인 경우가 많다. 깊이로 감점하면 오히려 손해다.

지금은 매번 전체를 다시 만든다. 858 청크에 몇 분 걸리므로 스케줄로
돌리기 전에 증분 갱신(파일 해시를 남겨 바뀐 것만 교체)이 먼저 필요하다.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.json"
CHROMA = ROOT / "chroma_db"
COLLECTION = "homelab"

# 맥북(Mac M3). 공유기에서 static 으로 고정한 주소다. litellm 도 같은
# 곳을 본다. 임베딩은 프록시를 거칠 이유가 없어 직접 붙는다.
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://192.168.0.44:11434")

# bge-m3 는 다국어 모델이다. 이 코퍼스는 문서도 커밋 메시지도 한국어라
# 영어 중심인 nomic-embed-text 로는 검색이 잘 걸리지 않는다.
EMBED_MODEL = os.environ.get("RAG_EMBED", "bge-m3")

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

md_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)

# 헤더로 나눈 뒤에도 섹션 하나가 너무 길면 다시 자른다. 헤더 분할만으로는
# 크기 상한이 없다.
size_capper = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

# HCL 과 YAML 은 최상위 블록에서 잘라야 "이게 무슨 리소스였는지" 가 청크
# 안에 남는다. 앞의 구분자부터 우선 시도하고, 안 되면 뒤로 물러난다.
infra_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\nresource ",
        "\nmodule ",
        "\ndata ",
        "\nvariable ",
        "\noutput ",
        "\nlocals ",
        "\n---\n",  # YAML 문서 경계
        "\n\n",
        "\n",
        " ",
    ],
)

code_splitters = {
    ".py": RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=1500, chunk_overlap=200
    ),
    ".html": RecursiveCharacterTextSplitter.from_language(
        Language.HTML, chunk_size=1500, chunk_overlap=200
    ),
}

MARKDOWN_CATEGORIES = {"docs", "context", "history"}


def breadcrumb(entry: dict, headers: list[str]) -> str:
    """청크 본문 맨 앞에 붙일 출처 경로."""
    trail = [f"{entry['source']}/{entry['path'].split('/', 1)[-1]}"]
    trail += headers
    return " > ".join(trail)


def split_markdown(text: str, entry: dict) -> list[Document]:
    out = []
    for section in md_splitter.split_text(text):
        # 메타데이터 키가 h1..h4 라 정렬하면 문서 상의 순서와 같아진다.
        headers = [section.metadata[k] for k in sorted(section.metadata) if section.metadata[k]]
        crumb = breadcrumb(entry, headers)
        for piece in size_capper.split_text(section.page_content):
            out.append(
                Document(
                    page_content=f"{crumb}\n\n{piece}",
                    metadata={
                        "depth": len(headers),
                        "heading": headers[-1] if headers else "",
                        "breadcrumb": crumb,
                    },
                )
            )
    return out


def split_plain(text: str, entry: dict, splitter) -> list[Document]:
    crumb = breadcrumb(entry, [])
    return [
        Document(
            page_content=f"{crumb}\n\n{piece}",
            metadata={"depth": 0, "heading": "", "breadcrumb": crumb},
        )
        for piece in splitter.split_text(text)
    ]


def chunk(entry: dict) -> list[Document]:
    text = (DATA / entry["path"]).read_text(encoding="utf-8")
    category = entry["category"]

    if category in MARKDOWN_CATEGORIES:
        docs = split_markdown(text, entry)
    elif category == "app-code":
        docs = split_plain(text, entry, code_splitters.get(entry["ext"], size_capper))
    else:
        docs = split_plain(text, entry, infra_splitter)

    # 출처 정보는 전부 스칼라로 넣는다. Chroma 가 리스트를 못 받는다.
    for d in docs:
        d.metadata.update(
            source=entry["source"],
            category=category,
            path=entry["path"],
            origin=entry["origin"],
            ext=entry["ext"],
        )
    return docs


def build() -> None:
    if not MANIFEST.exists():
        raise SystemExit("manifest.json 이 없다. collect_docs.py 를 먼저 돌려라.")

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))

    chunks: list[Document] = []
    for entry in entries:
        chunks.extend(chunk(entry))

    by_cat: dict[str, int] = {}
    for d in chunks:
        by_cat[d.metadata["category"]] = by_cat.get(d.metadata["category"], 0) + 1

    print(f"문서 {len(entries)} 개 → 청크 {len(chunks)} 개")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12} {v:5} 청크")

    # 임베딩 모델이 바뀌면 기존 벡터와 차원도 의미도 안 맞는다. 섞이지
    # 않도록 통째로 비우고 다시 만든다.
    if CHROMA.exists():
        shutil.rmtree(CHROMA)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA),
    )

    # 한 번에 넣으면 네트워크 너머 맥북이 다 끝날 때까지 아무 출력이 없다.
    # 나눠 넣어야 어디까지 갔는지 보인다.
    batch = 64
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i : i + batch])
        print(f"  임베딩 {min(i + batch, len(chunks)):5}/{len(chunks)}", flush=True)

    print(f"\n인덱스 완료 → {CHROMA} (collection: {COLLECTION}, embed: {EMBED_MODEL})")


if __name__ == "__main__":
    build()
