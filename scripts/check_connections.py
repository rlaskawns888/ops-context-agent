"""외부 연결 점검: Azure OpenAI(채팅·임베딩) / Langfuse / Neo4j / Postgres.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/check_connections.py

설정은 `.env`에서 읽는다 (common.settings). 비용이 드는 호출은 최소 토큰으로 1회씩만 한다.
모든 항목이 OK면 exit 0, 하나라도 FAIL이면 exit 1.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from common.settings import Settings, get_settings

PING = "ping"
CHAT_MAX_COMPLETION_TOKENS = 16


class MissingConfigError(Exception):
    pass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _require(settings: Settings, *fields: str) -> None:
    missing = [f.upper() for f in fields if not getattr(settings, f)]
    if missing:
        raise MissingConfigError(f".env 미설정: {', '.join(missing)}")


def _secret(value) -> str:
    return value.get_secret_value()


def _langfuse(settings: Settings):
    _require(settings, "langfuse_host", "langfuse_public_key", "langfuse_secret_key")
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=_secret(settings.langfuse_secret_key),
        base_url=settings.langfuse_host,
    )


def _azure_client(settings: Settings):
    _require(settings, "azure_openai_endpoint", "azure_openai_api_key", "azure_openai_api_version")
    # Langfuse 래퍼 클라이언트: 이 클라이언트로 한 호출은 자동으로 Langfuse에 기록된다.
    from langfuse.openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=_secret(settings.azure_openai_api_key),
        api_version=settings.azure_openai_api_version,
    )


def check_chat(settings: Settings) -> str:
    _require(settings, "azure_openai_chat_deployment")
    langfuse = _langfuse(settings)
    client = _azure_client(settings)

    response = client.chat.completions.create(
        name="check_connections.chat",
        model=settings.azure_openai_chat_deployment,
        messages=[{"role": "user", "content": PING}],
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )
    langfuse.flush()

    usage = response.usage
    tokens = f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else "usage 없음"
    return f"model={response.model}, {tokens}"


def check_langfuse(settings: Settings) -> str:
    langfuse = _langfuse(settings)
    if not langfuse.auth_check():
        raise RuntimeError("auth_check 실패 (키 또는 host 확인)")
    return f"인증 OK, 트레이스 확인: {settings.langfuse_host} → Tracing (name=check_connections.*)"


def check_embedding(settings: Settings) -> str:
    _require(settings, "azure_openai_embedding_deployment")
    langfuse = _langfuse(settings)
    client = _azure_client(settings)

    response = client.embeddings.create(
        name="check_connections.embedding",
        model=settings.azure_openai_embedding_deployment,
        input=PING,
    )
    langfuse.flush()
    return f"dim={len(response.data[0].embedding)}"


def neo4j_driver_config(settings: Settings) -> tuple[str, str | None]:
    """드라이버에 넘길 (URI, 신뢰할 CA 파일 경로)를 정한다.

    - NEO4J_TRUSTED_CA 없음: URI 그대로 (neo4j+s:// — OS 신뢰 저장소로 검증)
    - NEO4J_TRUSTED_CA 있음: '+s'를 뗀 스킴 + encrypted=True + TrustCustomCAs
      (neo4j+s:// 와 trusted_certificates는 함께 쓸 수 없다)
    - 검증을 끄는 '+ssc' 스킴은 허용하지 않는다.
    """
    uri = settings.neo4j_uri
    scheme, sep, rest = uri.partition("://")
    if not sep:
        raise ValueError(f"NEO4J_URI 형식 오류: {uri!r}")
    if scheme.endswith("+ssc"):
        raise ValueError("인증서 검증을 끄는 '+ssc' 스킴은 사용하지 않는다 (NEO4J_TRUSTED_CA 사용)")

    ca = settings.neo4j_trusted_ca
    if not ca:
        return uri, None
    if not Path(ca).is_file():
        raise FileNotFoundError(f"NEO4J_TRUSTED_CA 파일 없음: {ca}")
    return f"{scheme.removesuffix('+s')}://{rest}", ca


def check_neo4j(settings: Settings) -> str:
    _require(settings, "neo4j_uri", "neo4j_user", "neo4j_password")
    uri, trusted_ca = neo4j_driver_config(settings)
    from neo4j import GraphDatabase, TrustCustomCAs

    tls = (
        {"encrypted": True, "trusted_certificates": TrustCustomCAs(trusted_ca)}
        if trusted_ca
        else {}
    )
    auth = (settings.neo4j_user, _secret(settings.neo4j_password))
    with GraphDatabase.driver(uri, auth=auth, connection_timeout=10, **tls) as driver:
        driver.verify_connectivity()  # 재시도 없이 즉시 실패시킨다 (execute_query는 ~30초 재시도)
        records, _, _ = driver.execute_query("RETURN 1 AS n")
    value = records[0]["n"]
    if value != 1:
        raise RuntimeError(f"RETURN 1 결과가 {value!r}")
    via = f"TrustCustomCAs({trusted_ca})" if trusted_ca else "OS 신뢰 저장소"
    return f"verify_connectivity OK, RETURN 1 → 1 ({uri.partition('://')[0]}://, {via})"


def check_postgres(settings: Settings) -> str:
    _require(settings, "database_url")
    import psycopg

    with (
        psycopg.connect(_secret(settings.database_url), connect_timeout=5) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("vector 확장이 설치되어 있지 않음")
    return f"vector {row[0]}"


CHECKS: list[tuple[str, Callable[[Settings], str]]] = [
    ("Azure OpenAI 채팅", check_chat),
    ("Langfuse", check_langfuse),
    ("Azure OpenAI 임베딩", check_embedding),
    ("Neo4j", check_neo4j),
    ("Postgres", check_postgres),
]


def run_checks(settings: Settings) -> list[CheckResult]:
    results = []
    for name, check in CHECKS:
        try:
            results.append(CheckResult(name, True, check(settings)))
        except Exception as exc:  # 항목별로 실패 사유를 출력하고 다음 항목을 계속 점검한다
            reason = str(exc).splitlines()[0][:300] if str(exc) else ""
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {reason}"))
    return results


def main() -> int:
    results = run_checks(get_settings())
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{'OK' if r.ok else 'FAIL':4}] {r.name:<{width}}  {r.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
