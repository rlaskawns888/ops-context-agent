# 빌드 컨텍스트: 레포 루트 (docker compose가 지정)
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---- builder: uv로 이 서비스와 의존성만 .venv에 설치 ----
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 1) 의존성 레이어 (소스 변경 시 캐시 재사용)
COPY pyproject.toml uv.lock ./
COPY packages/common/pyproject.toml packages/common/
COPY services/agent-api/pyproject.toml services/agent-api/
COPY services/mcp-server/pyproject.toml services/mcp-server/
COPY services/ops-api/pyproject.toml services/ops-api/
COPY services/ingest/pyproject.toml services/ingest/
RUN --mount=type=cache,target=/root/.cache/uv     uv sync --frozen --no-dev --package ops-api --no-install-workspace

# 2) 워크스페이스 패키지 (common + 이 서비스)
COPY packages/common packages/common
COPY services/ops-api services/ops-api
RUN --mount=type=cache,target=/root/.cache/uv     uv sync --frozen --no-dev --package ops-api --no-editable

# ---- runtime: .venv만 복사 ----
FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 1000 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
WORKDIR /app
EXPOSE 8002
CMD ["uvicorn", "ops_api.main:app", "--host", "0.0.0.0", "--port", "8002"]
