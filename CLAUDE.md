# CLAUDE.md

사내 IT 운영 Context Agent. Slack에서 지식 Q&A와 업무 처리(MCP Tool)를 수행한다.
배경·목표·단계 계획은 [docs/project-context.md](docs/project-context.md), 확정 설계는 [docs/design.md](docs/design.md), 실험 기록은 [experiments.md](experiments.md).

## 스택

| 영역 | 결정 |
|---|---|
| 언어 | Python(주력), Cypher, SQL, TypeScript |
| API | FastAPI (모든 서비스) |
| Agent | LangGraph |
| 구조화 출력 | Pydantic |
| MCP | MCP Python SDK |
| 채널 | Slack (Slack Bolt for Python, 무료 워크스페이스) |
| 벡터 DB | PostgreSQL + pgvector |
| 그래프 DB | Neo4j AuraDB Free |
| 한국어 BM25 | rank_bm25 + kiwipiepy |
| LLM | Azure OpenAI 상위 모델 (저가 모델로 품질 낮추지 않음) |
| 임베딩 | text-embedding-3-large vs bge-m3 비교 실험 |
| 리랭커 | bge-reranker-v2-m3 |
| 관측·평가 | Langfuse |
| 프론트 | React + TypeScript (Vite) |
| 인프라 | Docker Compose, GitHub Actions |
| 클라우드 | Azure (Container Apps, Static Web Apps, ACR) — 학생 크레딧 사용 |
| 예산 | API 비용 월 1만 원 이내 (학생 크레딧으로 충당), 모델 품질은 유지 |
| 제외 | Spring Boot 사용 안 함, AWS 사용 안 함, 게임 도메인 제외 |

## 구조

```
services/
  agent-api/    src/agent_api/, tests/
  mcp-server/   src/mcp_server/, tests/
  ops-api/      src/ops_api/, tests/
  ingest/       src/ingest/, tests/
packages/common/  서비스 공용 코드 (src/common/)
data/raw/         합성 원천 데이터
data/generated/   생성 산출물 (gitignore)
eval/             평가셋·평가 스크립트
scripts/          일회성·운영 스크립트
docs/             project-context.md, design.md
```

uv workspace(루트 `pyproject.toml`), Python 3.12 고정. ruff·pytest 설정은 루트에만 둔다.

## 서비스 (Docker Compose)

1. **agent-api**: FastAPI + LangGraph 오케스트레이터, Slack 연동, 검색, 메모리
2. **mcp-server**: 업무 시스템 API를 감싸는 MCP 툴 서버 (계정·티켓·권한)
3. **ops-api**: mock 업무 시스템 (계정, 티켓, 권한 신청) — FastAPI + PostgreSQL 별도 스키마
4. **ingest**: 지식 적재 파이프라인 (배치 실행)
5. **admin-web**: 관리자 콘솔 (React) — 7단계에서 추가

**Agent는 업무 시스템을 직접 호출하지 않고 반드시 MCP를 거친다.**
agent-api 코드에서 ops-api를 HTTP로 직접 호출하거나 ops-api 패키지를 import하지 않는다.

## 규칙

- **기능 1개 단위로 작업한다.** 지시받은 기능 범위 밖의 코드는 건드리지 않는다.
- **시크릿 하드코딩 금지.** 키·토큰·접속 정보는 `.env`에서만 읽는다. 새 키가 필요하면 `.env.example`에 이름만 추가한다.
- **LLM 출력은 Pydantic으로 검증한다.** 자유 텍스트를 파싱해서 쓰지 않는다.
- **LLM 호출은 전부 Langfuse로 추적한다.** 추적되지 않는 LLM 호출 경로를 만들지 않는다.
- **평가 수치를 임의로 만들지 않는다.** 실제 실행 결과만 보고하고, 실행하지 않았으면 실행하지 않았다고 말한다. `experiments.md`는 사용자가 직접 작성한다.
- **테스트 없는 기능은 완료로 보지 않는다.** 기능마다 해당 서비스의 `tests/`에 테스트를 추가하고 통과를 확인한다.

## 실행 명령

```bash
uv sync --all-packages       # 의존성 설치 (워크스페이스 멤버 전체 포함; 그냥 uv sync는 루트 dev 도구만 설치)
docker compose up -d         # 서비스 기동 (compose 파일은 이후 단계에서 추가)
uv run pytest                # 전체 테스트
uv run pytest services/agent-api   # 서비스 단위 테스트
uv run ruff check .          # 린트
uv run ruff format .         # 포맷
```
