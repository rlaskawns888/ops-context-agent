"""Neo4j 연결 없이 scripts/neo4j_init.cypher 와 문장 분리 로직을 검증한다."""

import importlib.util
import re
from pathlib import Path

from common.schema.knowledge import EntityType

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location("neo4j_init", _SCRIPTS / "neo4j_init.py")
neo4j_init = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(neo4j_init)

_UNIQUE = re.compile(r"FOR\s*\(\s*\w+\s*:\s*(\w+)\s*\)\s*REQUIRE\s+\w+\.(\w+)\s+IS\s+UNIQUE", re.I)


def _statements() -> list[str]:
    return neo4j_init.split_statements(neo4j_init.CYPHER_FILE.read_text(encoding="utf-8"))


def test_every_create_has_if_not_exists() -> None:
    creates = [s for s in _statements() if s.upper().startswith("CREATE")]

    assert creates
    assert len(creates) == len(_statements())
    for stmt in creates:
        assert re.search(r"\bIF\s+NOT\s+EXISTS\b", stmt, re.I), stmt


def test_unique_constraint_labels_match_entity_types_and_source() -> None:
    unique = [m.groups() for s in _statements() if (m := _UNIQUE.search(s))]
    labels = {label for label, _ in unique}

    assert labels == {t.value for t in EntityType} | {"Source"}
    assert len(unique) == len(labels)  # 라벨당 하나
    for label, prop in unique:
        assert prop == ("source_id" if label == "Source" else "canonical_name"), label


def test_split_statements_ignores_comments_and_blank_lines() -> None:
    text = """
// 머리 주석; 세미콜론이 있어도 문장이 아니다

CREATE CONSTRAINT a IF NOT EXISTS
FOR (n:System) REQUIRE n.canonical_name IS UNIQUE;   // 줄 끝 주석

   // 들여쓴 주석
CREATE INDEX b IF NOT EXISTS FOR (n:Error) ON (n.code);
;

CREATE INDEX c IF NOT EXISTS FOR (n:Policy) ON (n.valid_to)
"""
    statements = neo4j_init.split_statements(text)

    assert len(statements) == 3
    assert statements[0] == (
        "CREATE CONSTRAINT a IF NOT EXISTS\nFOR (n:System) REQUIRE n.canonical_name IS UNIQUE"
    )
    assert all("//" not in s for s in statements)


def test_cypher_file_statement_count() -> None:
    assert len(_statements()) == 8 + 5  # 유니크 제약 8개 + 인덱스 5개


def test_format_table_aligns_columns() -> None:
    table = neo4j_init.format_table([("error_code_idx", "Error", "code")])

    header, rule, row = table.splitlines()
    assert header.split() == ["name", "labels", "properties"]
    assert row.split() == ["error_code_idx", "Error", "code"]
    assert header.index("labels") == row.index("Error")
