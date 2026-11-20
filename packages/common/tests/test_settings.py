import pytest

from common.settings import Settings


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == "postgresql://u:p@db:5432/x"
    assert "u:p" not in repr(settings)
