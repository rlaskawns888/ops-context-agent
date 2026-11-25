"""경계 강제: services/ 코드는 정답·생성 데이터를 읽지 않는다 (적재 입력은 data/raw만).

평가 코드(eval/)만 data/labels·data/gold에 접근할 수 있다. 문자열 검사라 경로를 쪼개 조립하는
우회까지 막지는 못한다 — 코드 리뷰에서 함께 본다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ROOT / "services"
FORBIDDEN = re.compile(r"data[/\\]+(labels|gold|generated)")
SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".txt", ".md", ".sh"}


def _source_files() -> list[Path]:
    out = []
    for p in SERVICES.rglob("*"):
        if not p.is_file() or SKIP_DIRS & set(p.relative_to(SERVICES).parts):
            continue
        if p.suffix in TEXT_SUFFIXES or p.name.startswith("Dockerfile"):
            out.append(p)
    return out


def find_violations(files: list[Path]) -> list[str]:
    hits = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{f.as_posix()}:{n}: {line.strip()}")
    return hits


def test_services_do_not_reference_labels_gold_or_generated():
    files = _source_files()
    assert any(f.suffix == ".py" for f in files)
    assert find_violations(files) == []


def test_detector_catches_forbidden_paths(tmp_path):
    bad = tmp_path / "x.py"
    lines = [
        'A = "data/labels/pii.jsonl"',
        'B = r"data\\gold\\knowledge"',
        'C = Path("data/generated")',
        'OK = "data/raw/slack.jsonl"',
    ]
    bad.write_text("\n".join(lines), encoding="utf-8")
    assert len(find_violations([bad])) == 3
