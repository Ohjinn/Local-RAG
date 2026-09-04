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

두 번째 실행부터는 바뀐 문서만 다시 임베딩한다. 파일 내용의 sha256 을
청크 메타데이터(`doc_hash`)에 같이 넣어 두고, 다음 실행 때 manifest 의
현재 해시와 비교해 새 문서·바뀐 문서만 지웠다 다시 넣는다. 해시를 별도
상태 파일이 아니라 인덱스 안에 두는 이유는, 상태 파일은 인덱스와 따로
놀다가 어긋나고 나중에 Chroma 를 k3s 로 옮길 때 따라가지도 않기 때문이다.

청킹 규칙 자체를 고쳤을 때는 파일이 안 바뀌었으니 해시도 같다. 그때는
`--full` 로 통째로 다시 만들어야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from chroma_store import chroma_kwargs, describe, LOCAL_DIR as CHROMA

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


def file_hash(path: Path) -> str:
    """파일 내용의 sha256. mtime 이 아니라 내용을 보는 이유는, collect_docs.py
    가 원본을 data/ 로 복사하면서 mtime 을 새로 찍기 때문이다. 내용이 그대로면
    다시 임베딩할 이유가 없다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def breadcrumb(entry: dict, headers: list[str]) -> str:
    """청크 본문 맨 앞에 붙일 출처 경로."""
    trail = [f"{entry['source']}/{entry['path'].split('/', 1)[-1]}"]
    trail += headers
    return " > ".join(trail)


# 마크다운 표의 구분선. `|---|---|` 또는 `| :--- | ---: |`
_TABLE_RULE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")


def table_headers_of(section: str) -> list[tuple[int, str]]:
    """절 안의 **모든** 표에 대해 (시작 위치, 머리글 두 줄) 목록을 만든다.

    처음에는 첫 번째 표의 머리글만 찾아 절 전체에 붙였다. 그게 틀렸다 —
    한 절에 표가 여럿이면 뒤쪽 표의 조각에 앞쪽 표의 머리글이 붙는다.
    실제로 디스크 목록 행에 `| OS 이미지 | 업무 구분 | 대상 |` 이 붙었다.
    **머리글이 없는 것보다 나쁘다.** 값의 뜻을 틀리게 알려 주기 때문이고,
    검색 거리도 0.834 에서 0.843 으로 오히려 밀렸다.

    그래서 위치까지 같이 들고, 조각마다 바로 앞의 것을 고른다.
    """
    out = []
    pos = 0
    lines = section.splitlines(keepends=True)
    starts = []
    for line in lines:
        starts.append(pos)
        pos += len(line)

    for i, line in enumerate(lines):
        if i == 0:
            continue
        if _TABLE_RULE.match(line.rstrip("\n")) and _TABLE_ROW.match(lines[i - 1]):
            header = lines[i - 1].rstrip() + "\n" + line.rstrip()
            out.append((starts[i - 1], header))
    return out


def header_for(piece: str, section: str, headers: list[tuple[int, str]]) -> str | None:
    """이 조각이 속한 표의 머리글을 고른다.

    조각의 첫 줄을 절 안에서 찾아 위치를 알아낸 뒤, 그보다 앞에 있는 표
    머리글 중 가장 가까운 것을 쓴다. 찾지 못하면 붙이지 않는다 — 틀리게
    붙이느니 안 붙이는 편이 낫다.
    """
    if not headers:
        return None
    probe = piece.lstrip().splitlines()[0] if piece.strip() else ""
    if not probe:
        return None
    at = section.find(probe)
    if at < 0:
        return None
    best = None
    for start, header in headers:
        if start <= at:
            best = header
        else:
            break
    return best


def restore_table_header(piece: str, header: str | None) -> str:
    """표 중간부터 시작하는 조각에 머리글을 되돌려 준다.

    긴 표가 청크 크기에 걸려 잘리면 **두 번째 조각부터는 머리글 행을 잃는다.**
    남는 것은 이런 모양이다 —

        | | 웹로그 수집 서버 #2 | | disk-prd-wcap-02-logs-01 | ... | 64 | ...

    값은 있는데 `64` 가 디스크 GB 인지 메모리인지 알 수 없다. 사람도 모르고
    임베딩도 모른다. 실제로 "운영계 WEB 서버 사양이 뭐야" 에 답하지 못했고,
    거리가 0.834 까지 밀렸다(잘 되는 질문은 0.490).

    머리글을 붙이면 두 가지가 같이 좋아진다. 사람이 읽을 수 있게 되고,
    **"디스크 크기" 같은 단어가 청크 안에 생겨서** 질문과 뜻이 이어진다.

    비용은 작다. 실측으로 머리글 평균 88 자, 대상 청크 평균 1,093 자라
    청크당 8%, 문서 전체로는 2.9% 다. **청크 개수는 늘지 않는다** — 새로
    만드는 게 아니라 기존 조각에 덧붙이는 것이라서 벡터 수도 HNSW 도 그대로다.
    """
    if not header:
        return piece
    stripped = piece.lstrip()
    # 표 행으로 시작하지 않으면 표 조각이 아니다.
    if not _TABLE_ROW.match(stripped):
        return piece
    # 이미 구분선을 갖고 있으면 머리글이 살아 있는 조각이다.
    if any(_TABLE_RULE.match(l) for l in piece.splitlines()):
        return piece
    return f"{header}\n{piece}"


def split_markdown(text: str, entry: dict) -> list[Document]:
    out = []
    for section in md_splitter.split_text(text):
        # 메타데이터 키가 h1..h4 라 정렬하면 문서 상의 순서와 같아진다.
        headers = [section.metadata[k] for k in sorted(section.metadata) if section.metadata[k]]
        crumb = breadcrumb(entry, headers)
        theads = table_headers_of(section.page_content)
        for piece in size_capper.split_text(section.page_content):
            piece = restore_table_header(
                piece, header_for(piece, section.page_content, theads)
            )
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


def chunk(entry: dict, doc_hash: str) -> list[Document]:
    text = (DATA / entry["path"]).read_text(encoding="utf-8")
    category = entry["category"]

    if category in MARKDOWN_CATEGORIES:
        docs = split_markdown(text, entry)
    elif category == "app-code":
        docs = split_plain(text, entry, code_splitters.get(entry["ext"], size_capper))
    else:
        docs = split_plain(text, entry, infra_splitter)

    # 출처 정보는 전부 스칼라로 넣는다. Chroma 가 리스트를 못 받는다.
    # doc_hash / embed_model 은 다음 실행 때 무엇을 다시 만들지 판단하는
    # 근거다. 인덱스가 곧 상태 파일 역할을 한다.
    for d in docs:
        d.metadata.update(
            source=entry["source"],
            category=category,
            path=entry["path"],
            origin=entry["origin"],
            ext=entry["ext"],
            doc_hash=doc_hash,
            embed_model=EMBED_MODEL,
        )
    return docs


def read_index(store: Chroma) -> tuple[dict[str, list[str]], dict[str, str], str]:
    """기존 인덱스에서 문서별 청크 id·해시와, 만들 때 쓴 임베딩 모델을 읽는다.

    벡터는 안 가져오고 메타데이터만 본다. 858 청크 기준 한순간이다.
    """
    got = store.get(include=["metadatas"])

    ids_by_path: dict[str, list[str]] = {}
    hash_by_path: dict[str, str] = {}
    embed_used = ""

    for cid, md in zip(got["ids"], got["metadatas"]):
        path = md.get("path", "")
        ids_by_path.setdefault(path, []).append(cid)
        hash_by_path[path] = md.get("doc_hash", "")
        embed_used = embed_used or md.get("embed_model", "")

    return ids_by_path, hash_by_path, embed_used


def build(full: bool = False) -> None:
    if not MANIFEST.exists():
        raise SystemExit("manifest.json 이 없다. collect_docs.py 를 먼저 돌려라.")

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # 빈 파일은 청크가 0 개라 인덱스에 아무 흔적도 남기지 못한다. 걸러내지
    # 않으면 매 실행마다 "새 문서" 로 잡혀 "바뀐 것 없음" 판정이 영영 나오지
    # 않는다. azure-standard 의 빈 outputs.tf 2 개가 실제로 그랬다.
    blank = {
        e["path"]
        for e in entries
        if not (DATA / e["path"]).read_text(encoding="utf-8").strip()
    }
    if blank:
        entries = [e for e in entries if e["path"] not in blank]
        print(f"빈 파일 {len(blank)} 개는 건너뛴다.")

    hashes = {e["path"]: file_hash(DATA / e["path"]) for e in entries}

    # --full 은 로컬과 원격에서 지우는 대상이 다르다. 로컬은 디렉토리를 통째로
    # 지우면 되지만, 원격에서 같은 짓을 하면 로컬 디렉토리만 지우고 서버의
    # 컬렉션은 그대로 남는다 — "전체 재생성" 이라 말하고 아무것도 안 지우는
    # 상황이 된다. 원격일 때는 컬렉션을 지운다.
    remote = "client" in chroma_kwargs()
    if full:
        if remote:
            import chromadb.errors
            tmp = Chroma(collection_name=COLLECTION, **chroma_kwargs())
            try:
                tmp.delete_collection()
                print("--full: 원격 컬렉션을 지웠다.")
            except Exception as e:
                print(f"--full: 지울 컬렉션이 없거나 실패({type(e).__name__}) — 계속한다.")
        elif CHROMA.exists():
            shutil.rmtree(CHROMA)
            print("--full: 기존 인덱스를 지웠다.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        **chroma_kwargs(),
    )

    ids_by_path, hash_by_path, embed_used = read_index(store)

    # 임베딩 모델이 바뀌면 기존 벡터와 차원도 의미도 안 맞는다. 섞어 두면
    # 검색이 조용히 이상해지므로 컬렉션째 버리고 다시 만든다. 디렉토리를
    # 지우지 않고 컬렉션을 지우는 건, 이미 열려 있는 sqlite 핸들을
    # 흔들지 않으면서 차원 제약까지 같이 털어내기 위해서다.
    if embed_used and embed_used != EMBED_MODEL:
        print(f"임베딩 모델이 {embed_used} → {EMBED_MODEL} 로 바뀌었다. 전체 재생성한다.")
        store.delete_collection()
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            **chroma_kwargs(),
        )
        ids_by_path, hash_by_path = {}, {}

    added = [e for e in entries if e["path"] not in hash_by_path]
    changed = [
        e
        for e in entries
        if e["path"] in hash_by_path and hash_by_path[e["path"]] != hashes[e["path"]]
    ]
    removed = [p for p in ids_by_path if p not in hashes]
    kept = len(entries) - len(added) - len(changed)

    print(f"문서 {len(entries)} 개 — 새로 {len(added)}, 변경 {len(changed)}, "
          f"삭제 {len(removed)}, 그대로 {kept}")

    if not added and not changed and not removed:
        print("바뀐 것이 없다. 임베딩을 건너뛴다.")
        print(f"인덱스 유지 → {describe()} (청크 {len(store.get(include=[])['ids'])} 개)")
        return

    # 바뀐 문서는 옛 청크를 먼저 지운다. 청크 경계가 달라지면 새로 넣는
    # 것만으로는 옛 청크가 그대로 남아 검색에 계속 걸린다.
    stale = [cid for e in changed for cid in ids_by_path[e["path"]]]
    stale += [cid for p in removed for cid in ids_by_path[p]]
    if stale:
        store.delete(ids=stale)
        print(f"  옛 청크 {len(stale)} 개 삭제")

    chunks: list[Document] = []
    for entry in added + changed:
        chunks.extend(chunk(entry, hashes[entry["path"]]))

    if chunks:
        by_cat: dict[str, int] = {}
        for d in chunks:
            by_cat[d.metadata["category"]] = by_cat.get(d.metadata["category"], 0) + 1
        print(f"  새 청크 {len(chunks)} 개")
        for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12} {v:5} 청크")

        # 한 번에 넣으면 네트워크 너머 맥북이 다 끝날 때까지 아무 출력이
        # 없다. 나눠 넣어야 어디까지 갔는지 보인다.
        batch = 64
        for i in range(0, len(chunks), batch):
            store.add_documents(chunks[i : i + batch])
            print(f"  임베딩 {min(i + batch, len(chunks)):5}/{len(chunks)}", flush=True)

    total = len(store.get(include=[])["ids"])
    print(f"\n인덱스 완료 → {describe()} "
          f"(collection: {COLLECTION}, embed: {EMBED_MODEL}, 청크 {total} 개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full",
        action="store_true",
        help="인덱스를 지우고 전부 다시 만든다. 청킹 규칙을 고쳤을 때 쓴다.",
    )
    build(full=parser.parse_args().full)
