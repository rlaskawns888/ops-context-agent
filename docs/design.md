# 설계 문서 (확정 설계)

> 단계별로 확정된 설계만 기록한다. 실험 과정과 판단 이유는 `experiments.md`에 둔다.
> 마지막 갱신: 2026-11-25 / 완료 단계: 0, 1, 2

---

## 1. 개요

Slack에서 동작하는 사내 IT 운영 Agent.
- 지식 Q&A: VPN, MFA, 계정 잠금, 보안 프로그램 오류, 권한 정책 등 문의에 근거와 함께 답변
- 업무 처리: 계정 잠금 해제, 티켓 생성·조회, 권한 신청을 MCP Tool로 실행 (실행 전 사용자 승인)
- 정책 검증(Guardrail): 권한 신청을 지식그래프로 검증
- 이관(Human-in-the-loop): 근거 부족·위험 요청은 담당자에게 대화 요약과 함께 이관
- 피드백 루프: 👍/👎와 담당자 수정 답변을 Context DB에 반영
- 지식의 원천: Slack 대화·티켓·회의록·문서에서 추출한 지식

---

## 2. 기술 스택

| 영역 | 결정 |
|---|---|
| 언어 | Python(주력), Cypher, SQL, TypeScript |
| API | FastAPI (모든 서비스) |
| Agent | LangGraph |
| 구조화 출력 | Pydantic |
| MCP | MCP Python SDK |
| 채널 | Slack (Slack Bolt for Python) |
| 벡터 DB | PostgreSQL + pgvector |
| 그래프 DB | Neo4j AuraDB Free |
| 한국어 BM25 | rank_bm25 + kiwipiepy |
| LLM | Azure OpenAI gpt-5.4-mini |
| 임베딩 | text-embedding-3-large vs bge-m3 비교 실험 |
| 리랭커 | bge-reranker-v2-m3 |
| 관측·평가 | Langfuse Cloud |
| 프론트 | React + TypeScript (Vite) |
| 인프라 | Docker Compose, GitHub Actions |
| 클라우드 | Azure (Container Apps, Static Web Apps, ACR) |

### 서비스 구성
| 서비스 | 역할 | 포트 |
|---|---|---|
| agent-api | FastAPI + LangGraph 오케스트레이터, Slack 연동, 검색, 메모리 | 8000 |
| mcp-server | 업무 시스템 API를 감싸는 MCP 툴 서버 (계정·티켓·권한) | 8001 |
| ops-api | mock 업무 시스템 (계정, 티켓, 권한 신청) | 8002 |
| ingest | 지식 적재 파이프라인 (배치) | 8003 |
| admin-web | 관리자 콘솔 (7단계에서 생성) | - |

Agent는 업무 시스템을 직접 호출하지 않고 반드시 MCP를 거친다.

---

## 3. 처리 흐름

- 지식 적재: 수집 → 정제(개인정보 마스킹, 중복 제거) → 청킹·임베딩 → LLM 지식 추출(Pydantic) → 정답과 비교 검증 → pgvector + Neo4j 적재
- 질의 처리: Slack 질문 → 의도 분류 → 사용자 메모리 조회 → 하이브리드 검색(벡터 + BM25 + 그래프 멀티홉) → 리랭킹 → 근거 포함 답변 → Guardrail → 응답
- 업무 처리: 업무 요청 → 그래프 정책 검증 → Slack 승인 버튼 → MCP Tool → ops-api 실행 → 결과 알림 (위험·근거 부족 시 담당자 이관)
- 피드백·배포: 👍/👎·담당자 수정 → 관리자 콘솔 검수 → Context DB 반영 → 평가셋 편입 / PR → 핵심 평가 30문항 → 기준 미달 시 merge 차단 → Azure 배포

---

## 4. 단계별 확정 결정

### 0. 준비 (완료 2026-11-20)
- 레포: 모노레포 `ops-context-agent`, uv workspace, Python 3.12 고정
- 구조: services/{agent-api, mcp-server, ops-api, ingest}, packages/common, data/{raw, generated}, eval/, scripts/, docs/design.md, experiments.md, CLAUDE.md
- 공통 코드: packages/common (pydantic-settings 설정 로더, 로깅, Langfuse·DB 클라이언트)
- PostgreSQL: pgvector/pgvector:pg16 컨테이너 1개, 스키마 분리 (knowledge / ops)
- Neo4j: AuraDB Free (로컬 컨테이너 없음)
- Langfuse: Langfuse Cloud 무료 플랜
- Azure OpenAI 배포: 채팅 = gpt-5.4-mini (2026-11-23 변경), 임베딩 = text-embedding-3-large
- CI: PR·main push에서 uv sync → ruff → pytest (평가 CI는 8단계에서 추가)
- 연결 확인: scripts/check_connections.py (Azure 채팅·임베딩, Neo4j, Postgres)

### 1. 지식 추출 스키마 설계 (완료 2026-11-20)
- 노드 8종: System, Error, Solution, Permission, UserGroup, Policy, Team, Source
  - 엔티티 공통 속성: canonical_name, aliases, evidence(원문 인용, 필수)
  - Source는 LLM이 뽑지 않고 적재 파이프라인이 붙임
- 관계 9종 (허용 방향 외 조합은 검증에서 거부)
  - Error -OCCURS_IN-> System, Error -RESOLVED_BY-> Solution, System -DEPENDS_ON-> System
  - Permission -ON-> System, Policy -GOVERNS-> Permission, Policy -APPLIES_TO-> UserGroup
  - Policy -SUPERSEDES-> Policy, System -OWNED_BY-> Team, (모든 노드) -EXTRACTED_FROM-> Source
- 설계 원칙
  - 암묵지: Solution.knowledge_type = tacit (별도 노드 없음)
  - 정책: 주체(UserGroup) / 대상(Permission) / 효과(ALLOW·DENY·REQUIRE_APPROVAL). Guardrail은 그래프 조회로 판정
  - 시간 지식: Policy·Solution에 valid_from / valid_to
  - 별명 통합: canonical_name + aliases
- 추출 출력: ExtractionResult(chunk_id, entities, relations, policies), 청크 1개당 1개
  - 관계가 가리키는 (타입, 이름)은 같은 결과의 entities에 존재해야 함
  - Azure OpenAI strict 구조화 출력 호환: 전 필드 required, 선택 값은 null 명시, 추가 필드 금지
  - 코드: packages/common schema/knowledge.py
- Neo4j: 엔티티 7종 canonical_name 유니크, Source.source_id 유니크 / 인덱스 Error.code, Solution.knowledge_type, Policy.valid_from·valid_to, Source.source_type
  - 실행: scripts/neo4j_init.py (멱등)
- 정답 지식(gold): 모델 schema/gold.py (knowledge.py 모델 재사용)
  - 유형 5종: tacit_fix, official_procedure, policy, policy_revision, multihop
  - render_as: 어떤 원문(slack, ticket, guide, minutes)으로 만들지 지정
  - gold의 evidence = 원문에 반드시 들어가야 할 핵심 문장
- 데이터 생성 방식: 정답 지식을 먼저 만들고 이를 바탕으로 원문 생성 → 추출 평가용 수동 라벨링 불필요
- 추출 정확도: 엔티티 (type, canonical_name) 일치, 관계 (source, relation_type, target) 일치 기준 정밀도·재현율
- 3단계로 넘긴 결정
  - 그래프 적재 시 정책은 policies 목록만 기준으로 사용
  - 정책 개정 시 이름 충돌은 canonical_name에 시행일 붙여 해결 (예: "외주 운영DB 쓰기 금지 (2026-01-01)")

### 2. 합성 데이터 생성 (완료 2026-11-25)
- 생성 순서: world → gold → mock 사용자 → gold 원문 렌더링 → 노이즈 문서 → PII 삽입·코퍼스 분리
  - 사용자를 원문보다 먼저 생성 (작성자·요청자가 실제 mock 사용자이고 소속 그룹이 정책과 일치해야 함)
- 세계관: data/world/world.yaml + schema/world.py, 검증 scripts/validate_world.py
  - 시스템 16개(AD 계정, SSO, MFA, VPN, 메일, 그룹웨어, 사내 포털, 티켓 시스템, DRM, EDR, 백신, 망연계, 운영DB, 개발DB, 형상관리, 배포시스템), 팀, 사용자 그룹, 권한, 오류코드
  - 이름 참조 무결성, canonical·alias 전역 중복 금지, depends_on 순환 금지
  - 모든 생성물은 world의 이름만 사용
- gold: K001~K060 (5유형 × 12), scripts/datagen/generate_gold.py, 리포트 gold_report.py
  - 검증: world 이름 존재, error code·OCCURS_IN 일치, 유형별 규칙, 날짜 범위, 중복 거부 → 실패 시 data/gold/_rejected/
  - 결과: 16개 시스템 전부 등장, 거부 1건 (world에 없는 권한명 생성)
- 날짜 규칙 (전 단계 공통)
  - valid_to는 마지막 날 포함(inclusive)
  - 정책 개정 시 이전 정책 valid_to = 신규 정책 valid_from - 1일
  - 유효 판정: valid_from <= 날짜 <= valid_to (valid_to null = 현재 유효)
  - 문서 날짜 범위 2025-01-01 ~ 2026-11-30
- mock 사용자: data/generated/users.jsonl 200명, scripts/datagen/generate_users.py
  - Faker ko_KR, seed=42 결정적 생성, LLM 미사용
  - it_staff 약 15명 (world 팀마다 2명 이상, 이관 대상), 나머지 requester
  - granted_permissions는 현재 유효한 gold DENY 정책 위반 금지
  - ops 스키마 DB 시드는 5단계에서 users.jsonl로 적재
- gold 원문 렌더링: scripts/datagen/render_docs.py → 105건 (slack 30, ticket 23, guide 32, minutes 20)
  - 회의록은 gold 24건을 20개에 묶음
  - gold evidence 문장이 원문에 그대로 포함되는지 검사 (정규화 후 부분 문자열)
  - gold에 없는 해결법·정책 생성 금지, 메타 정보(gold, K0 등) 노출 금지
  - slack·ticket은 world aliases 혼용, guide는 canonical_name
- 노이즈 문서: 755건 (slack 270, ticket 477, guide 8)
  - 원칙: 시나리오(시스템, 오류코드, 인물, 날짜, 처리 결과)는 코드가 결정(plan_noise.py → noise_plan.jsonl), LLM은 문장만 작성(render_noise.py)
  - 문의 유형 비율(가정값): 계정 잠금·비밀번호 20, VPN 15, MFA·SSO 15, 보안 프로그램 15, 권한 신청 15, 메일·그룹웨어 8, 망연계 7, 기타 5
  - noise_kind: general 70 (gold의 오류·정책 조합 사용 금지) / repeat 20 (gold 재문의, related_gold_ids 기록) / unresolved 10 (slack만)
  - 권한 신청 티켓 status는 작성일 기준 유효한 gold 정책으로 판정 (DENY → REJECTED)
- 생성 품질: 1차 검증 통과율 gold 원문 렌더링 91% (user 오류 16건) → 노이즈(시나리오 코드 결정) 98% (user 오류 0건)
- PII 삽입: scripts/datagen/inject_pii.py, LLM 미사용, seed=42
  - slack 40%, ticket 60% 문서에 phone, email, employee_no, internal_ip, pc_name 삽입 (작성자 실제 값)
  - 결과: 420개 문서, PII 642개
  - evidence 문장 내부는 수정하지 않음
- 코퍼스·라벨 분리 (정답 누수 구조적 차단)
  - data/raw/{slack,ticket,guide,minutes}.jsonl: 적재 입력 860건, 정답 관련 필드 제거
  - data/labels/doc_gold.jsonl (doc_id → gold_ids, related_gold_ids, noise_kind), data/labels/pii.jsonl (doc_id, field_path, pii_type, value)
  - services/ 코드는 data/raw만 읽음. data/labels·gold·generated 참조 시 테스트 실패. eval/만 정답 접근
- manifest: data/raw/manifest.json (모델, 토큰·비용, seed, 입력·프롬프트 SHA256, 문서·PII 수, 비율 가정값 표시)
- 비용 계산: 환경변수 LLM_PRICE_INPUT/CACHED_INPUT/OUTPUT_USD_PER_1M (공통), 추론 토큰은 출력에 포함
  - 생성 호출 기본 reasoning_effort=low
  - 2단계 총 비용 $2.3777 (gold $0.25, gold 원문 $0.35, 노이즈 $1.78)
- 3단계로 넘긴 결정
  - 마스킹 재현율은 data/labels/pii.jsonl 기준으로 측정
  - 추출 정밀도·재현율은 gold 원문(doc_gold.jsonl에 gold_ids가 있는 문서)으로 측정
  - 검색 평가 시 related_gold_ids 문서도 정답으로 인정 (4단계)
  - LLM 호출 네트워크 오류 재시도(지수 백오프 3회)를 packages/common 공통 클라이언트에 적용

### 3. 지식 적재 파이프라인
- (완료 후 기록)

## 3. 지식 적재 파이프라인

## 4. 하이브리드 검색·평가

## 5. Agent·MCP·Guardrail

## 6. Slack·피드백 루프

## 7. 관리자 콘솔

## 8. CI·배포·문서화
