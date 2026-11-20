"""Neo4j 초기 제약·인덱스 생성.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/neo4j_init.py

`neo4j_init.cypher`를 세미콜론 기준으로 나눠 한 문장씩 실행한다. 모든 문장이
IF NOT EXISTS라 여러 번 실행해도 안전하다. 데이터는 적재·삭제하지 않는다.
접속 설정은 common.settings(.env)에서 읽고, URI·TLS 처리는 check_connections를 재사용한다.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

CYPHER_FILE = Path(__file__).resolve().with_name("neo4j_init.cypher")


def split_statements(text: str) -> list[str]:
    """Cypher 텍스트를 문장 목록으로 나눈다. `//` 주석과 빈 문장은 버린다.

    문자열 리터럴 안의 `//`·`;`는 고려하지 않는다 (스키마 DDL 전용).
    """
    lines = [line.split("//", 1)[0].rstrip() for line in text.splitlines()]
    body = "\n".join(line for line in lines if line.strip())
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


def format_table(rows: Sequence[tuple[str, str, str]]) -> str:
    header = ("name", "labels", "properties")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(3)]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in [header, tuple("-" * w for w in widths), *rows]
    ]
    return "\n".join(lines)


def _join(values: list[str] | None) -> str:
    return ", ".join(values) if values else "-"


def _show(driver, query: str) -> list[tuple[str, str, str]]:
    records, _, _ = driver.execute_query(
        f"{query} YIELD name, labelsOrTypes, properties RETURN name, labelsOrTypes, properties "
        "ORDER BY name"
    )
    return [(r["name"], _join(r["labelsOrTypes"]), _join(r["properties"])) for r in records]


def main() -> int:
    from check_connections import _require, _secret, neo4j_driver_config
    from neo4j import GraphDatabase, TrustCustomCAs

    from common.settings import get_settings

    settings = get_settings()
    _require(settings, "neo4j_uri", "neo4j_user", "neo4j_password")
    uri, trusted_ca = neo4j_driver_config(settings)
    tls = (
        {"encrypted": True, "trusted_certificates": TrustCustomCAs(trusted_ca)}
        if trusted_ca
        else {}
    )
    auth = (settings.neo4j_user, _secret(settings.neo4j_password))

    statements = split_statements(CYPHER_FILE.read_text(encoding="utf-8"))
    with GraphDatabase.driver(uri, auth=auth, connection_timeout=10, **tls) as driver:
        driver.verify_connectivity()
        for i, stmt in enumerate(statements, 1):
            print(f"[{i}/{len(statements)}] {stmt.splitlines()[0]}")
            driver.execute_query(stmt)

        print("\n== SHOW CONSTRAINTS ==")
        print(format_table(_show(driver, "SHOW CONSTRAINTS")))
        print("\n== SHOW INDEXES ==")
        print(format_table(_show(driver, "SHOW INDEXES")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
