"""홈랩 문서를 data/ 로 모으고, 청킹에 쓸 메타데이터를 manifest.json 에 남긴다.

RAG 에 넣을 코퍼스를 만드는 첫 단계다. 원본을 그대로 두고 복사본을 만드는
이유는 두 가지다. 하나는 무엇이 인덱스에 들어갔는지 한눈에 보이게 하려는
것이고, 다른 하나는 시크릿이 섞여 들어가는 사고를 막기 위해서다.

RAG 에는 문서 단위 권한이 없다. 한 번 인덱스에 들어간 내용은 그 인덱스를
질의할 수 있는 누구에게나 노출된다. 그래서 "넣지 않는 것" 이 유일한 통제
수단이고, 그 결정을 이 파일 한 곳에 모아 둔다.

파일마다 category 를 붙인다. 문서 종류에 따라 잘라야 하는 방식이 다르기
때문이다. 마크다운은 헤더 경계에서 잘라야 문단이 온전하고, HCL 과 YAML 은
리소스 블록 경계에서 잘라야 "이 리소스가 뭐였는지" 가 청크 안에 남는다.
파이썬은 함수 경계다. 같은 크기로 일괄해서 자르면 셋 다 망가진다.
실제 분할은 build_index.py 가 이 manifest 를 읽어서 한다.

git 커밋 로그도 함께 뽑는다(category: history). 코드는 현재 상태만
보여주지만 커밋 메시지에는 "왜 그렇게 했는지" 와 "무엇을 하지 않기로
했는지" 가 남는다. 지금 파일 어디에도 없는 정보라 따로 긁어온다.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
OUT = Path(__file__).parent / "data"
MANIFEST = OUT / "manifest.json"

# 확장자로 카테고리를 정한다. 검색할 때 필터로도 쓴다. "왜 이렇게 짰지" 류
# 질문은 docs 에서, "그 포트 몇 번이지" 류는 infra-code 에서 답이 나온다.
CATEGORY_BY_EXT = {
    ".md": "docs",
    ".tf": "infra-code",
    ".hcl": "infra-code",
    ".yml": "infra-code",
    ".yaml": "infra-code",
    ".sh": "infra-code",
    ".py": "app-code",
    ".html": "app-code",
}

# 확장자가 없거나 확장자만으로는 성격이 안 드러나는 파일들.
CATEGORY_BY_NAME = {
    "Dockerfile": "infra-code",
    "requirements.txt": "app-code",
}

# 어디서든 걸리면 무조건 빼는 것들. 문서가 아니거나, 캐시거나, 빌드 산물이다.
ALWAYS_EXCLUDE = {".git/", "__pycache__", ".venv/", "node_modules/", ".terraform/"}


@dataclass
class Source:
    alias: str
    root: Path
    exts: set[str]
    names: set[str] = field(default_factory=set)  # 확장자 없는 파일을 이름으로 지정
    exclude: set[str] = field(default_factory=set)  # 경로에 이 문자열이 있으면 제외
    category: str | None = None  # 지정하면 확장자 매핑을 무시하고 전부 이 카테고리
    git_log: bool = False  # git 커밋 로그를 별도 문서로 뽑을지


SOURCES = [
    Source(
        alias="homelab-iac",
        root=HOME / "git_code/homelab-iac",
        exts={".md", ".tf", ".sh", ".yml", ".yaml"},
        # terraform.tfvars 에는 Proxmox API 토큰과 러너 비밀번호가 실제 값으로
        # 들어 있다. git 추적 대상도 아니다. 절대 넣지 않는다.
        exclude={"terraform.tfvars"},
        git_log=True,
    ),
    Source(
        alias="homelab-gitops",
        root=HOME / "git_code/homelab-gitops",
        exts={".md", ".yml", ".yaml", ".py"},
        names={"requirements.txt"},
        # pumpyeong 은 없앨 서비스다. 곧 사라질 것을 인덱스에 넣으면 "떠 있다"
        # 고 답하게 된다. 레포에서 매니페스트를 실제로 지우면 이 줄도 지운다.
        exclude={"pumpyeong"},
        git_log=True,
    ),
    Source(
        alias="wizparking",
        root=HOME / "git_code/wizparking",
        exts={".md", ".py", ".html"},
        names={"Dockerfile", "requirements.txt"},
        git_log=True,
    ),
    Source(
        alias="azure-standard",
        root=HOME / "git_code/azure-terraform-standard",
        exts={".md", ".tf"},
        exclude={"terraform.tfvars", "secrets"},
        git_log=True,
    ),
    Source(
        alias="context",
        root=HOME / ".claude",
        exts={".md"},
        # 전역 컨텍스트와 메모만 가져온다. 세션 로그나 설정은 문서가 아니다.
        # MEMORY.md 는 다른 메모로 가는 목차일 뿐이라 본문이 중복된다.
        exclude={
            "todos",
            "shell-snapshots",
            "statsig",
            "plugins",
            "cache",
            "MEMORY.md",
        },
        category="context",
    ),
]

# 아래는 의도적으로 넣지 않는다. 지우면 다시 들어오므로 이유를 남겨 둔다.
#
#   bs4/webapp/          wizparking 의 구버전이다. app.py 가 616 줄 대 729 줄,
#                        parking_core.py 는 19 줄 차이. 둘 다 넣으면 검색이
#                        구버전 청크를 물어와 옛날 코드로 답하게 된다.
#   ~/investwells/       gitops 안 k8s/base/investwells/src/ 와 바이트 단위로
#                        같다. 실제 배포되는 gitops 쪽이 정본이다. 게다가
#                        investwells_partners.{json,xlsx} 업체 자료가 섞여 있다.
#   pumpyeong 콘텐츠     k3s 노드의 /opt/homelab/pumpyeong 에만 두기로 이미
#                        정한 업체 자료다. 그 결정을 여기서도 지킨다.
#   wiamachine_azure/    고객사 실무 코드. 넣을지는 아직 정하지 않았다.

# 파일 내용에 이런 게 보이면 통째로 건너뛴다. 확장자 필터를 빠져나온 것을
# 잡는 마지막 그물이다. 값이 실제로 박힌 경우만 걸리도록, 참조 표현
# (secretKeyRef, os.environ, var.foo 등)은 제외한다.
SECRET_VALUE = re.compile(
    r"""(?ix)
    (password|secret|token|api[_-]?key|private[_-]?key)
    \s*[:=]\s*["']?
    (?!.*(secretKeyRef|secretName|os\.environ|var\.|\$\{|<|xxx|your-|example|change))
    [A-Za-z0-9_/+=-]{16,}
    """
)
PRIVATE_KEY_BLOCK = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def wanted(path: Path, src: Source) -> bool:
    return path.suffix in src.exts or path.name in src.names


def excluded(rel: str, src: Source) -> bool:
    return any(p in rel for p in ALWAYS_EXCLUDE | src.exclude)


def categorize(path: Path, src: Source) -> str:
    if src.category:
        return src.category
    return CATEGORY_BY_NAME.get(path.name) or CATEGORY_BY_EXT.get(path.suffix, "docs")


# 레코드는 NUL, 필드는 US(0x1f) 로 나눈다. 커밋 본문에는 개행이 자유롭게
# 들어가므로 줄 단위로 자를 수 없다. --name-only 가 붙여주는 파일 목록은
# 마지막 필드 뒤에 그대로 따라온다.
GIT_FORMAT = "%x00%h%x1f%ad%x1f%an%x1f%s%x1f%b%x1f"


def collect_git_log(src: Source) -> str | None:
    """커밋 하나가 청크 하나가 되도록 마크다운으로 뽑는다.

    제목을 `##` 로 두는 게 핵심이다. build_index 의 헤더 분할기가 이
    경계에서 자르므로, 청크가 커밋 단위로 딱 떨어진다. 크기로 자르면
    한 커밋의 본문이 두 청크로 찢기거나 남의 커밋과 섞인다.
    """
    if not (src.root / ".git").exists():
        return None

    try:
        raw = subprocess.run(
            ["git", "log", "--date=short", f"--format={GIT_FORMAT}", "--name-only"],
            cwd=src.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  git log 실패({src.alias}): {e}")
        return None

    out = [f"# {src.alias} 커밋 로그\n"]
    count = 0
    for record in raw.split("\x00"):
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) < 5:
            continue
        sha, date, author, subject, body = parts[:5]
        files = [f for f in parts[5].splitlines() if f.strip()] if len(parts) > 5 else []

        # 커밋 메시지에 시크릿이 실려 있으면 그 커밋만 뺀다. 로그 전체를
        # 버리면 나머지 멀쩡한 이력까지 잃는다.
        if SECRET_VALUE.search(record) or PRIVATE_KEY_BLOCK.search(record):
            continue

        out.append(f"\n## {sha} {subject}\n")
        out.append(f"- 날짜: {date}")
        out.append(f"- 작성자: {author}")
        if files:
            # 변경 파일을 같이 남겨야 "이 파일이 왜 이렇게 됐지" 라는
            # 질문이 해당 커밋에 걸린다. 파일명이 곧 검색 표면이다.
            out.append(f"- 변경 파일: {', '.join(files)}")
        if body.strip():
            out.append(f"\n{body.strip()}")
        count += 1

    if not count:
        return None
    print(f"  git log: {src.alias} 커밋 {count} 개")
    return "\n".join(out) + "\n"


def collect() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries, skipped = [], []
    for src in SOURCES:
        if not src.root.exists():
            print(f"  건너뜀(경로 없음): {src.root}")
            continue

        for path in sorted(src.root.rglob("*")):
            if not path.is_file() or not wanted(path, src):
                continue

            rel = path.relative_to(src.root)
            if excluded(str(rel), src):
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if SECRET_VALUE.search(text) or PRIVATE_KEY_BLOCK.search(text):
                skipped.append(f"{src.alias}/{rel}")
                continue

            # 원본 디렉토리 구조를 그대로 재현한다. 이름을 뭉개지 않으니
            # data/ 를 열어보는 것만으로 무엇이 들어갔는지 알 수 있다.
            #
            # 예전처럼 본문 맨 앞에 "# 출처:" 를 끼워넣지 않는다. 그 h1 이
            # 마크다운 헤더 분할에서 모든 문서의 최상위 헤더가 되어버려
            # 실제 문서 구조를 덮어쓰기 때문이다. 출처는 아래 manifest 로
            # 넘기고, build_index.py 가 청크마다 메타데이터로 붙인다.
            dest = OUT / src.alias / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")

            entries.append(
                {
                    "path": str(dest.relative_to(OUT)),
                    "source": src.alias,
                    "category": categorize(path, src),
                    "origin": str(path),
                    "ext": path.suffix or path.name,
                }
            )

        if src.git_log and (log := collect_git_log(src)):
            dest = OUT / src.alias / "_git_log.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(log, encoding="utf-8")
            entries.append(
                {
                    "path": str(dest.relative_to(OUT)),
                    "source": src.alias,
                    "category": "history",
                    "origin": f"git -C {src.root} log",
                    "ext": ".md",
                }
            )

    MANIFEST.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n수집 완료: {len(entries)} 개 → {OUT}")

    by_cat: dict[str, int] = {}
    by_src: dict[str, int] = {}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        by_src[e["source"]] = by_src.get(e["source"], 0) + 1

    print("\n카테고리별:")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12} {v:4} 개")
    print("\n출처별:")
    for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16} {v:4} 개")

    if skipped:
        print(f"\n시크릿이 보여 제외한 파일 {len(skipped)} 개:")
        for s in skipped:
            print(f"  - {s}")

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\n총 용량: {total / 1024:.0f} KB")


if __name__ == "__main__":
    collect()
