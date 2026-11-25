# data/labels — 평가 전용 정답 라벨

> **평가 전용. ingest·agent 코드(services/ 하위 전부)에서 읽기 금지.**
> 적재 입력은 `data/raw`만 사용한다. 이 폴더는 `eval/`의 평가 코드만 읽는다.
> `scripts/tests/test_data_boundary.py`가 services/ 소스에 이 경로 문자열이 있으면 실패시킨다.

`scripts/datagen/inject_pii.py`가 `data/raw`와 함께 생성한다 (seed 42, 수정 금지 — 재생성만).

| 파일 | 한 줄 | 용도 |
|---|---|---|
| `doc_gold.jsonl` | `{doc_id, gold_ids, related_gold_ids, noise_kind, issue_category}` | 문서 ↔ gold 지식 정답, 노이즈 종류, 문의 분류 |
| `pii.jsonl` | `{doc_id, field_path, pii_type, value}` | 3단계 마스킹 검증 정답 (삽입한 가짜 개인정보 위치) |

- `gold_ids`: 이 문서가 담고 있는 gold 지식 id (`data/gold/knowledge/*.yaml`). 노이즈 문서는 빈 배열.
- `related_gold_ids`: 재문의(repeat) 노이즈가 다시 묻는 gold id.
- `noise_kind`: `general` / `repeat` / `unresolved`, gold 원문은 `null`.
- `issue_category`: 노이즈 문서의 문의 분류 (가정 비율로 배정, 실제 통계 아님). gold 원문은 `null`.
- `field_path`: `data/raw` 레코드 안의 위치. 예: `messages[3].text`, `description`, `comments[1].text`.
- `pii_type`: `phone`, `email`, `employee_no`, `internal_ip`, `pc_name`. 값은 모두 합성 값이다.
