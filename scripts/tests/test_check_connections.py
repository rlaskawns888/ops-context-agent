"""네트워크 없이 검증 가능한 부분만 테스트한다: 설정 누락 시 항목별 FAIL과 사유."""

import importlib.util
import sys
from pathlib import Path

import pytest

from common.settings import Settings

_SCRIPT = Path(__file__).resolve().parents[1] / "check_connections.py"
_spec = importlib.util.spec_from_file_location("check_connections", _SCRIPT)
check_connections = importlib.util.module_from_spec(_spec)
sys.modules["check_connections"] = check_connections  # dataclass가 모듈을 조회하므로 등록 필요
_spec.loader.exec_module(check_connections)


@pytest.fixture
def empty_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    return Settings(_env_file=None)


def test_all_checks_fail_with_missing_config_reason(empty_settings: Settings) -> None:
    results = check_connections.run_checks(empty_settings)

    assert [r.name for r in results] == [name for name, _ in check_connections.CHECKS]
    assert all(not r.ok for r in results)
    assert all("MissingConfigError" in r.detail for r in results)


def test_missing_config_reason_names_env_keys(empty_settings: Settings) -> None:
    results = {r.name: r for r in check_connections.run_checks(empty_settings)}

    assert "NEO4J_URI" in results["Neo4j"].detail
    assert "DATABASE_URL" in results["Postgres"].detail
    assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in results["Azure OpenAI 채팅"].detail


def _neo4j_settings(uri: str, trusted_ca: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        neo4j_uri=uri,
        neo4j_user="u",
        neo4j_password="p",
        neo4j_trusted_ca=trusted_ca,
    )


def test_neo4j_without_trusted_ca_keeps_uri(empty_settings: Settings) -> None:
    uri, ca = check_connections.neo4j_driver_config(
        _neo4j_settings("neo4j+s://x.databases.neo4j.io")
    )

    assert uri == "neo4j+s://x.databases.neo4j.io"
    assert ca is None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("neo4j+s://x.databases.neo4j.io", "neo4j://x.databases.neo4j.io"),
        ("bolt+s://x.databases.neo4j.io:7687", "bolt://x.databases.neo4j.io:7687"),
        ("neo4j://localhost:7687", "neo4j://localhost:7687"),
    ],
)
def test_neo4j_with_trusted_ca_drops_plus_s(
    empty_settings: Settings, tmp_path: Path, given: str, expected: str
) -> None:
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("dummy")

    uri, ca = check_connections.neo4j_driver_config(_neo4j_settings(given, str(ca_file)))

    assert uri == expected
    assert ca == str(ca_file)


def test_neo4j_trusted_ca_missing_file_fails(empty_settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NEO4J_TRUSTED_CA"):
        check_connections.neo4j_driver_config(
            _neo4j_settings("neo4j+s://x", str(tmp_path / "none.pem"))
        )


def test_neo4j_rejects_ssc_scheme(empty_settings: Settings) -> None:
    with pytest.raises(ValueError, match="ssc"):
        check_connections.neo4j_driver_config(_neo4j_settings("neo4j+ssc://x"))
