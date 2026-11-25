# 설계 (확정 사항)

> 각 단계가 끝나면 확정된 설계만 기록한다.

# 사내 IT 운영 Context Agent — 설계 문서

> 단계별로 확정된 설계만 기록한다. 실험 과정과 판단 이유는 experiments.md에 남긴다.

## 0. 준비 · 개발 환경

### 0.1 레포 구조
- 모노레포 `ops-context-agent`, uv workspace, Python 3.12 고정

    ops-context-agent/
    ├─ services/
    │  ├─ agent-api/    # FastAPI + LangGraph 오케스트레이터, Slack 연동, 검색, 메모리
    │  ├─ mcp-server/   # 업무 시스템 API를 감싸는 MCP 툴 서버 (계정·티켓·권한)
    │  ├─ ops-api/      # mock 업무 시스템 (계정, 티켓, 권한 신청)
    │  └─ ingest/       # 지식 적재 파이프라인 (배치)
    ├─ packages/common/ # 공통 코드
    ├─ data/{raw, generated}/
    ├─ eval/
    ├─ scripts/
    ├─ docs/design.md
    ├─ experiments.md
    └─ CLAUDE.md

- admin-web(React)은 7단계에서 생성

### 0.2 공통 코드 (packages/common)
- pydantic-settings 기반 설정 로더
- 로깅
- Langfuse · DB 클라이언트

### 0.3 서비스 포트
| 서비스 | 포트 |
|---|---|
| agent-api | 8000 |
| mcp-server | 8001 |
| ops-api | 8002 |
| ingest | 8003 |

- Agent는 업무 시스템(ops-api)을 직접 호출하지 않고 반드시 mcp-server를 거친다.

### 0.4 저장소
- PostgreSQL: pgvector/pgvector:pg16 컨테이너 1개, 스키마 분리
  - knowledge: 지식 청크·임베딩
  - ops: mock 업무 시스템 데이터
- Neo4j: AuraDB Free (로컬 컨테이너 없음)

### 0.5 외부 서비스
- Azure OpenAI
  - 채팅: (배포 모델명 기입)
  - 임베딩: text-embedding-3-large
- Langfuse: Cloud 무료 플랜 (self-host 안 함, 데이터가 전부 합성이라 문제없음)

### 0.6 CI
- PR · main push 시: uv sync → ruff → pytest
- 평가 CI(핵심 30문항)는 8단계에서 추가

### 0.7 연결 확인
- scripts/check_connections.py: Azure 채팅·임베딩, Neo4j, Postgres 연결 점검
- Langfuse 트레이스 수신 확인

## 1. 지식 추출 스키마

### 1.1 목적
Slack 스레드·티켓·회의록·가이드 문서에서 LLM이 지식을 뽑을 때 쓰는 양식과,
그 지식을 저장할 그래프 구조를 정의한다.

### 1.2 노드 (8종)
| 라벨 | 의미 | 예 | 고유 속성 |
|---|---|---|---|
| System | 시스템·보안 프로그램 | VPN, MFA, 운영DB | description |
| Error | 오류 코드·증상 | E-102 | code, symptom |
| Solution | 해결 절차 | PC 재부팅 | steps, knowledge_type, valid_from, valid_to |
| Permission | 시스템별 권한 수준 | 운영DB WRITE | system_name, level |
| UserGroup | 고용 형태·직무 | 외주, 개발자 | kind |
| Policy | 업무 규칙 | 외주 운영DB 쓰기 금지 | valid_from, valid_to |
| Team | 담당 부서 | 보안운영팀 | channel |
| Source | 지식의 출처 원문 | Slack 스레드 | source_id, source_type |

- 엔티티 공통 속성: canonical_name(대표 이름), aliases(다른 이름), evidence(원문 인용)
- Source는 LLM이 뽑지 않고 적재 파이프라인이 붙인다.

### 1.3 관계 (9종)
| 관계 | 방향 | 의미 |
|---|---|---|
| OCCURS_IN | Error → System | 이 오류는 이 시스템에서 발생 |
| RESOLVED_BY | Error → Solution | 이 오류는 이 방법으로 해결 |
| DEPENDS_ON | System → System | 이 시스템은 저 시스템에 의존 |
| ON | Permission → System | 이 권한은 이 시스템의 권한 |
| GOVERNS | Policy → Permission | 이 규칙은 이 권한에 대한 규칙 |
| APPLIES_TO | Policy → UserGroup | 이 규칙은 이 집단에 적용 |
| SUPERSEDES | Policy → Policy | 새 규칙이 옛 규칙을 대체 |
| OWNED_BY | System → Team | 이 시스템의 담당 팀 (이관 대상) |
| EXTRACTED_FROM | (모든 노드) → Source | 이 지식이 나온 원문 (파이프라인이 생성) |

- 위 방향 외의 조합은 검증 단계에서 거부한다.

### 1.4 설계 원칙
- 암묵지: 별도 노드 없이 Solution.knowledge_type = tacit 으로 표시
  (공식 문서에 없고 대화에서만 나온 해결법)
- 정책: 주체(UserGroup) / 대상(Permission) / 효과(ALLOW·DENY·REQUIRE_APPROVAL) 3칸 구조.
  Guardrail은 LLM 판단이 아니라 그래프 조회로 판정한다.
- 시간 지식: Policy·Solution에 valid_from / valid_to 를 미리 둔다.
- 별명 통합: canonical_name + aliases 로 "VPN / 사내 VPN" 같은 표기를 하나로 묶는다.
- 근거 필수: evidence(원문 인용)가 없는 추출 결과는 거부한다.

### 1.5 추출 출력 형식 (ExtractionResult, 청크 1개당 1개)
- chunk_id, entities, relations, policies
- 검증: relations가 가리키는 (타입, 이름)은 같은 결과의 entities에 존재해야 함
- Azure OpenAI 구조화 출력(strict) 호환: 모든 필드 required, 선택 값은 null 명시, 추가 필드 금지
- 코드: packages/common/.../schema/knowledge.py

### 1.6 Neo4j 제약·인덱스
- 유니크: 엔티티 7종 canonical_name, Source.source_id
- 인덱스: Error.code, Solution.knowledge_type, Policy.valid_from/valid_to, Source.source_type
- 실행: uv run python scripts/neo4j_init.py (여러 번 실행해도 안전)

### 1.7 3단계로 넘긴 결정
- 정책 정보는 entities·relations와 policies 두 곳에 나온다.
  그래프 적재 시에는 policies 목록만 기준으로 사용한다.
- 정책 개정 시 이름 충돌(유니크 제약)은 canonical_name에 시행일을 붙여 해결한다.
  (예: "외주 운영DB 쓰기 금지 (2026-01-01)")

### 1.8 추출 정확도 측정
- 엔티티: (type, canonical_name) 일치 시 정답
- 관계: (source, relation_type, target) 일치 시 정답
- 정밀도(지어낸 게 없나)와 재현율(빠뜨린 게 없나)을 따로 기록
- 정답은 2단계에서 정답 지식그래프(YAML)를 먼저 만들고 그것으로 데이터를 생성해서 확보한다.
- 정답 지식 파일: data/gold/knowledge/K001~K005.yaml, 모델 schema/gold.py

## 2. 합성 데이터 생성

## 3. 지식 적재 파이프라인

## 4. 하이브리드 검색·평가

## 5. Agent·MCP·Guardrail

## 6. Slack·피드백 루프

## 7. 관리자 콘솔

## 8. CI·배포·문서화
