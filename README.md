# ops-context-agent

Slack에서 동작하는 사내 IT 운영 Context Agent — 흩어진 대화·티켓·문서에서 지식을 추출해 근거 있는 답변과 MCP 기반 업무 처리를 수행한다.

> **이 레포의 모든 데이터(문서, Slack 대화, 티켓, 회의록, 사용자)는 전부 합성 데이터입니다.** 실제 회사 내부 정보는 포함되어 있지 않습니다.

## 데이터

모든 데이터는 LLM으로 생성한 **합성 데이터**이며 실제 회사 정보는 없습니다. 사람 이름·사번·전화번호·메일·IP·PC명도 전부 가짜 값입니다.

### 생성 방식

```
world → gold → 원문 렌더링 → 노이즈 → PII 삽입
```

1. **world** (`data/world/world.yaml`): 가상 회사의 시스템·팀·사용자 그룹·권한·오류 코드 정의
2. **gold** (`scripts/datagen/generate_gold.py` → `data/gold/knowledge/`): 정답 지식 60건 (엔티티·관계·정책 + 근거 문장)
3. **원문 렌더링** (`render_docs.py`): gold를 Slack 스레드·티켓·가이드·회의록으로 풀어 씀
4. **노이즈** (`plan_noise.py` → `render_noise.py`): gold와 무관한 일반 문의·재문의·미해결 문서
5. **PII 삽입·분리** (`inject_pii.py`, LLM 호출 없음, seed 42): slack 40%·ticket 60% 문서에 작성자의 가짜 개인정보 문장을 덧붙이고, 적재 입력(`data/raw`)과 평가 정답(`data/labels`)을 폴더로 나눔

문의 유형(category) 비율은 **실제 통계가 아닌 가정값**입니다.

| 폴더 | 내용 | 읽는 곳 |
|---|---|---|
| `data/raw/` | 최종 코퍼스 + `manifest.json` (정답 필드 없음) | ingest·agent |
| `data/labels/` | 문서별 gold 정답, PII 위치 정답 | 평가(`eval/`)만 |
| `data/gold/` | gold 지식 원본 | 평가(`eval/`)만 |
| `data/generated/` | 중간 산출물 (gitignore) | 생성 스크립트만 |

### 규모

| 유형 | 문서 수 | PII 삽입 문서 |
|---|---:|---:|
| slack | 300 | 120 |
| ticket | 500 | 300 |
| guide | 40 | 0 |
| minutes | 20 | 0 |
| **합계** | **860** | **420** (PII 값 642개) |

### 생성 비용 (`data/raw/manifest.json`)

모델 `gpt-5.4-mini` (Azure OpenAI 배포), reasoning_effort `low`. dry-run·재시도 호출 포함 전체 합산.

| 단계 | 호출 | 입력 토큰 (캐시) | 출력 토큰 (추론) | 비용 (USD) |
|---|---:|---:|---:|---:|
| gold | 18 | 169,403 (88,832) | 43,482 (8,553) | 0.2512 |
| render_docs | 116 | 157,969 (3,840) | 51,173 (7,708) | 0.3462 |
| render_noise | 398 | 686,121 (35,072) | 286,540 (24,869) | 1.7803 |
| **합계** | **532** | **1,013,493 (127,744)** | **381,195 (41,130)** | **2.3777** |

gold 단계의 1회 호출은 단가 설정 전에 실행돼 비용 기록이 없어 합계에서 빠져 있습니다 (토큰은 포함).
