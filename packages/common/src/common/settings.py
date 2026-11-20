"""서비스 공통 설정 로더.

값은 환경변수 → `.env` 순서로 읽는다. 시크릿은 코드에 기본값으로 두지 않는다.
각 기능이 필요로 하는 값의 필수 여부는 그 기능을 구현할 때 검증한다.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"

    # PostgreSQL
    database_url: SecretStr | None = None

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Neo4j
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: SecretStr | None = None
    neo4j_trusted_ca: str | None = None  # 루트 CA 파일 경로 (선택)

    # Langfuse
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
