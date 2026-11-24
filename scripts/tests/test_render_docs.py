"""gold → 원문 렌더링: 계획·검증·재시도 (LLM 호출 없음)."""

import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import render_docs as rd
from common.schema.documents import Guide, Minutes, SlackThread
from common.schema.gold import load_gold_knowledge
from common.schema.knowledge import SourceType
from common.schema.world import load_world

WORLD = load_world(rd.WORLD_FILE)
GOLDS = load_gold_knowledge(rd.GOLD_DIR)
GOLD = {g.id: g for g in GOLDS}
USERS = rd.load_users(rd.USERS_FILE)
USERS_BY_ID = {u.user_id: u for u in USERS}
SLACK_TO_USER = {u.slack_user_id: u.user_id for u in USERS}


def _job(source_type: SourceType, *gold_roles: tuple[str, rd.Role]) -> rd.Job:
    return rd.Job(source_type, tuple(rd.Part(GOLD[gid], role) for gid, role in gold_roles))


def _ctx(job: rd.Job, doc_date: date | None = None) -> rd.JobContext:
    ctx = rd.JobContext.build(job, USERS, WORLD)
    if doc_date is not None:
        ctx.doc_date = doc_date
        ctx.created_at = datetime.combine(doc_date, ctx.created_at.time())
    return ctx


def _slack_draft(ctx: rd.JobContext, texts: list[str], day: date | None = None) -> rd.SlackDraft:
    day = day or ctx.doc_date
    people = ctx.cast.everyone
    start = datetime.combine(day, datetime.min.time()) + timedelta(hours=10)
    return rd.SlackDraft(
        title="VPN 접속 문의",
        channel="#it-helpdesk",
        messages=[
            {
                "slack_user_id": people[i % len(people)].slack_user_id,
                "ts": start + timedelta(minutes=5 * i),
                "text": text,
            }
            for i, text in enumerate(texts)
        ],
    )


def _slack_doc(ctx: rd.JobContext, texts: list[str], day: date | None = None) -> SlackThread:
    doc = rd.build_document(ctx, _slack_draft(ctx, texts, day), 1, SLACK_TO_USER)
    assert isinstance(doc, SlackThread)
    return doc


def _validate(doc: Any, job: rd.Job) -> list[str]:
    return rd.validate_document(doc, job, WORLD, set(USERS_BY_ID))


def _has(errors: list[str], kind: str) -> bool:
    return any(e.startswith(f"[{kind}]") for e in errors)


K001_SLACK = _job(SourceType.SLACK, ("K001", "full"))
K001_EVIDENCE = rd.required_evidence(GOLD["K001"])


# ---------------------------------------------------------------------------
# evidence 포함 검사
# ---------------------------------------------------------------------------


def test_evidence_all_included_passes() -> None:
    ctx = _ctx(K001_SLACK)
    # 앞뒤 말·줄바꿈·연속 공백이 섞여도 정규화 후 부분 문자열이면 통과
    first, *rest = K001_EVIDENCE
    texts = [f"안녕하세요ㅠㅠ\n{first.replace(' ', '  ', 1)}", *rest]
    doc = _slack_doc(ctx, texts)
    assert _validate(doc, K001_SLACK) == []


def test_missing_evidence_fails() -> None:
    ctx = _ctx(K001_SLACK)
    doc = _slack_doc(ctx, [*K001_EVIDENCE[:-1], "감사합니다!"])
    errors = _validate(doc, K001_SLACK)
    assert _has(errors, "evidence")
    assert K001_EVIDENCE[-1] in errors[0]


def test_paraphrased_evidence_fails() -> None:
    ctx = _ctx(K001_SLACK)
    changed = K001_EVIDENCE[2].replace("재부팅하고", "재시작하고")
    doc = _slack_doc(ctx, [*K001_EVIDENCE[:2], changed])
    assert _has(_validate(doc, K001_SLACK), "evidence")


def test_evidence_split_across_messages_fails() -> None:
    ctx = _ctx(K001_SLACK)
    sentence = K001_EVIDENCE[0]
    half = len(sentence) // 2
    doc = _slack_doc(ctx, [sentence[:half], sentence[half:], *K001_EVIDENCE[1:]])
    assert _has(_validate(doc, K001_SLACK), "evidence")


def test_check_evidence_normalizes_whitespace() -> None:
    assert rd.check_evidence(["가 나\n\n다  라"], ["가 나 다 라"]) == []
    assert rd.check_evidence(["가나다라"], ["가 나 다 라"]) == ["가 나 다 라"]


# ---------------------------------------------------------------------------
# 메타 정보·등장인물·이름·오류 코드
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leak",
    [
        "이건 K001 건이랑 같아요",
        "gold 데이터 기준으로",
        "evidence 문장입니다",
        "scenario상",
        "암묵지예요",
    ],
)
def test_meta_exposure_fails(leak: str) -> None:
    ctx = _ctx(K001_SLACK)
    doc = _slack_doc(ctx, [*K001_EVIDENCE, leak])
    assert _has(_validate(doc, K001_SLACK), "meta")


def test_meta_pattern_ignores_similar_words() -> None:
    assert rd.check_meta(["SK1234 장비, 오늘 K-100 회의실"]) == []


def test_unknown_user_fails() -> None:
    ctx = _ctx(K001_SLACK)
    draft = _slack_draft(ctx, K001_EVIDENCE)
    draft.messages[1].slack_user_id = "SZZZZZZZZ"
    doc = rd.build_document(ctx, draft, 1, SLACK_TO_USER)
    assert _has(_validate(doc, K001_SLACK), "user")


def test_error_code_not_in_gold_fails() -> None:
    ctx = _ctx(K001_SLACK)
    doc = _slack_doc(ctx, [*K001_EVIDENCE, "혹시 E-301도 같이 뜨나요?"])
    assert _has(_validate(doc, K001_SLACK), "error_code")


def test_system_not_mentioned_fails() -> None:
    job = _job(SourceType.SLACK, ("K001", "full"))
    ctx = _ctx(job)
    draft = _slack_draft(ctx, ["접속이 안 돼요", "재부팅 해보세요", "됐습니다"])
    draft.title = "접속 문의"
    errors = _validate(rd.build_document(ctx, draft, 1, SLACK_TO_USER), job)
    assert _has(errors, "name") and _has(errors, "evidence")


def _guide(ctx: rd.JobContext, body: str) -> Guide:
    draft = rd.GuideDraft(title="안내", version="1.0", body=body)
    doc = rd.build_document(ctx, draft, 1, SLACK_TO_USER)
    assert isinstance(doc, Guide)
    return doc


def _guide_body(sentences: list[str]) -> str:
    return "## 개요\n" + "\n".join(sentences[:1]) + "\n\n## 절차\n" + "\n".join(sentences[1:])


def test_guide_requires_sections_and_canonical_name() -> None:
    job = _job(SourceType.GUIDE, ("K004", "new"))
    ctx = _ctx(job, date(2026, 2, 2))
    evidence = job.required_evidence()
    assert _validate(_guide(ctx, _guide_body(evidence)), job) == []
    assert _has(_validate(_guide(ctx, "\n".join(evidence)), job), "format")


# ---------------------------------------------------------------------------
# policy_revision: evidence 분할·날짜 규칙
# ---------------------------------------------------------------------------


def test_revision_roles_split_minutes_old_guide_new() -> None:
    k004 = GOLD["K004"]
    assert rd.revision_role(k004, SourceType.MINUTES) == "old"
    assert rd.revision_role(k004, SourceType.GUIDE) == "new"
    assert rd.revision_role(GOLD["K001"], SourceType.SLACK) == "full"

    old_name, new_name = rd.revision_policy_names(k004)
    old_ev, new_ev = rd.required_evidence(k004, "old"), rd.required_evidence(k004, "new")
    supersedes = next(r.evidence for r in k004.relations if r.relation_type == "SUPERSEDES")
    old_policy = next(p.evidence for p in k004.policies if p.name == old_name)
    new_policy = next(p.evidence for p in k004.policies if p.name == new_name)
    assert old_policy in old_ev and new_policy not in old_ev and supersedes not in old_ev
    assert new_policy in new_ev and supersedes in new_ev and old_policy not in new_ev
    # 공통 사실(UserGroup·System·Permission)은 양쪽 모두
    shared = next(e.evidence for e in k004.entities if e.type == "Permission")
    assert shared in old_ev and shared in new_ev
    assert set(old_ev) | set(new_ev) == set(rd.required_evidence(k004))


def test_revision_date_windows() -> None:
    k004 = GOLD["K004"]  # 이전 2025-01-01~2025-12-31, 신규 2026-01-01~
    assert rd.date_window(k004, "old") == (date(2025, 1, 1), date(2025, 12, 31))
    assert rd.date_window(k004, "new") == (date(2026, 1, 1), rd.DATE_MAX)
    k040 = GOLD["K040"]  # 신규 2025-07-01
    assert rd.date_window(k040, "old")[1] == date(2025, 6, 30)


def test_new_policy_doc_dated_before_valid_from_fails() -> None:
    job = _job(SourceType.GUIDE, ("K004", "new"))
    body = _guide_body(job.required_evidence())
    assert not _has(_validate(_guide(_ctx(job, date(2026, 1, 1)), body), job), "date")
    assert _has(_validate(_guide(_ctx(job, date(2025, 12, 31)), body), job), "date")


def _minutes(ctx: rd.JobContext, discussion: list[str]) -> Minutes:
    draft = rd.MinutesDraft(
        title="해외 VPN 운영 점검 회의",
        agenda=["해외 VPN 접속 운영 현황"],
        discussion="\n".join(discussion),
        decisions=["현행 운영 유지"],
        action_items=[
            {"owner_user_id": ctx.cast.staff[0].user_id, "task": "현황 공유", "due_date": None}
        ],
    )
    doc = rd.build_document(ctx, draft, 1, SLACK_TO_USER)
    assert isinstance(doc, Minutes)
    return doc


def test_old_policy_doc_dated_after_new_valid_from_fails() -> None:
    job = _job(SourceType.MINUTES, ("K004", "old"))
    evidence = job.required_evidence()
    assert _validate(_minutes(_ctx(job, date(2025, 10, 1)), evidence), job) == []
    assert _has(_validate(_minutes(_ctx(job, date(2026, 1, 5)), evidence), job), "date")


def test_old_policy_doc_must_not_contain_new_policy() -> None:
    job = _job(SourceType.MINUTES, ("K004", "old"))
    leak = rd.forbidden_evidence(GOLD["K004"], "old")
    assert leak
    doc = _minutes(_ctx(job, date(2025, 10, 1)), [*job.required_evidence(), leak[0]])
    assert _has(_validate(doc, job), "date")


def test_policy_valid_from_limits_window() -> None:
    k038 = GOLD["K038"]  # 정책 valid_from 2025-02-01
    assert rd.date_window(k038)[0] == date(2025, 2, 1)
    assert rd.date_window(GOLD["K001"]) == (rd.DATE_MIN, rd.DATE_MAX)


# ---------------------------------------------------------------------------
# 작업 계획: minutes 배치
# ---------------------------------------------------------------------------


def test_minutes_plan_places_all_24_gold_into_20_docs() -> None:
    minutes_gold = [g.id for g in GOLDS if SourceType.MINUTES in g.render_as]
    assert len(minutes_gold) == 24
    groups = rd.plan_minutes(GOLDS, WORLD)
    assert len(groups) == rd.MINUTES_DOC_COUNT
    placed = [g.id for grp in groups for g in grp]
    assert sorted(placed) == sorted(minutes_gold)  # 빠짐·중복 없음
    pairs = [grp for grp in groups if len(grp) == 2]
    assert len(pairs) == 4 and all(1 <= len(grp) <= 2 for grp in groups)
    for a, b in pairs:
        same_system = rd.gold_systems(a) & rd.gold_systems(b)
        same_team = set(rd.responsible_teams(a, WORLD)) & set(rd.responsible_teams(b, WORLD))
        assert same_system or same_team
        roles = [rd.revision_role(g, SourceType.MINUTES) for g in (a, b)]
        start, end = rd.intersect_windows(
            rd.date_window(g, r) for g, r in zip((a, b), roles, strict=True)
        )
        assert start <= end


def test_minutes_plan_rejects_impossible_count() -> None:
    with pytest.raises(ValueError):
        rd.plan_minutes(GOLDS, WORLD, doc_count=5)
    with pytest.raises(ValueError):
        rd.plan_minutes(GOLDS, WORLD, doc_count=30)


def test_plan_jobs_covers_every_gold_render_pair_once() -> None:
    jobs = rd.plan_jobs(GOLDS, WORLD)
    pairs = Counter((gid, j.source_type.value) for j in jobs for gid in j.gold_ids)
    expected = {(g.id, st.value) for g in GOLDS for st in g.render_as}
    assert set(pairs) == expected and set(pairs.values()) == {1}
    assert Counter(j.source_type.value for j in jobs)["minutes"] == rd.MINUTES_DOC_COUNT
    for job in jobs:
        start, end = job.window
        assert start <= end, job.key


def test_pending_jobs_skips_already_generated(tmp_path: Path) -> None:
    jobs = rd.plan_jobs(GOLDS, WORLD)
    minutes_pair = next(j for j in jobs if j.source_type == "minutes" and len(j.gold_ids) == 2)
    ctx = _ctx(K001_SLACK)
    rd.append_document(_slack_doc(ctx, K001_EVIDENCE), tmp_path)
    # 회의록은 묶인 gold 중 하나만 있으면 아직 미완료
    line = {"doc_id": "MN-003", "gold_ids": minutes_pair.gold_ids[:1]}
    rd.output_file(SourceType.MINUTES, tmp_path).write_text(json.dumps(line) + "\n", "utf-8")

    done, next_number = rd.existing_state(tmp_path)
    assert next_number[SourceType.SLACK] == 2 and next_number[SourceType.MINUTES] == 4
    assert next_number[SourceType.GUIDE] == 1
    pending = rd.pending_jobs(jobs, done)
    keys = {j.key for j in pending}
    assert K001_SLACK.key not in keys and minutes_pair.key in keys
    assert len(pending) == len(jobs) - 1
    assert [j.source_type for j in rd.pending_jobs(jobs, done, SourceType.GUIDE, 3)] == [
        SourceType.GUIDE
    ] * 3


# ---------------------------------------------------------------------------
# 등장인물·날짜·스키마
# ---------------------------------------------------------------------------


def test_cast_policy_requester_in_subject_group_and_staff_in_owner_team() -> None:
    k003 = GOLD["K003"]  # 외주 → 운영DB, 담당 DB운영팀
    group = k003.policies[0].subject_group
    owner = next(s.owner_team for s in WORLD.systems if s.canonical_name == "운영DB")
    for st in (SourceType.TICKET, SourceType.GUIDE):
        ctx = _ctx(_job(st, ("K003", "full")))
        assert all(u.user_group == group for u in ctx.cast.requesters)
        assert ctx.cast.staff and all(u.it_team == owner for u in ctx.cast.staff)


def test_job_context_is_deterministic_and_inside_window() -> None:
    for job in rd.plan_jobs(GOLDS, WORLD):
        a, b = rd.JobContext.build(job, USERS, WORLD), rd.JobContext.build(job, USERS, WORLD)
        assert (a.doc_date, a.created_at, a.cast) == (b.doc_date, b.created_at, b.cast)
        start, end = job.window
        assert start <= a.doc_date <= end
        if job.source_type in (SourceType.MINUTES,):
            assert len(a.cast.everyone) >= 2


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("source_type", list(rd.SOURCE_TYPES))
def test_response_schema_is_strict_compatible(source_type: SourceType) -> None:
    schema = rd.response_json_schema(source_type)
    text = json.dumps(schema)
    for keyword in ('"oneOf"', '"const"', '"discriminator"', '"format"', '"pattern"'):
        assert keyword not in text
    for node in _walk(schema):
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


def test_prompt_has_no_gold_id_or_title() -> None:
    for job in rd.plan_jobs(GOLDS, WORLD)[:20]:
        messages = rd.build_messages(rd.JobContext.build(job, USERS, WORLD), WORLD)
        text = "\n".join(m["content"] for m in messages)
        for part in job.parts:
            assert part.gold.id not in text and part.gold.title not in text
        assert "tacit" not in text


# ---------------------------------------------------------------------------
# 재시도·거부 (가짜 클라이언트)
# ---------------------------------------------------------------------------


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=40,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
    )


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return _response(self.contents.pop(0))


def _run(client: FakeClient, ctx: rd.JobContext) -> tuple[rd.JobResult, list[rd.Attempt]]:
    logged: list[rd.Attempt] = []
    result = rd.render_job(
        client, "dep", ctx, WORLD, USERS_BY_ID, 7, log_call=lambda _u, a: logged.append(a)
    )
    return result, logged


def test_render_job_retries_with_failure_reasons() -> None:
    ctx = _ctx(K001_SLACK)
    bad = _slack_draft(ctx, [*K001_EVIDENCE[:2], "재부팅 해보세요"]).model_dump_json()
    good = _slack_draft(ctx, K001_EVIDENCE).model_dump_json()
    client = FakeClient(["{oops", bad, good])
    result, logged = _run(client, ctx)

    assert result.accepted and result.document.doc_id == "SL-0007"
    assert result.document.gold_ids == ["K001"]
    assert [a.evidence_ok for a in logged] == [None, False, True]
    assert _has(logged[0].reasons, "schema") and _has(logged[1].reasons, "evidence")
    retry_prompt = client.calls[2]["messages"][-1]["content"]
    assert K001_EVIDENCE[2] in retry_prompt
    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_render_job_rejects_after_max_retries(tmp_path: Path) -> None:
    ctx = _ctx(K001_SLACK)
    bad = _slack_draft(ctx, ["안녕하세요", "VPN 문의", "감사합니다"]).model_dump_json()
    client = FakeClient([bad] * (rd.MAX_RETRIES + 1))
    result, logged = _run(client, ctx)
    assert not result.accepted and len(logged) == rd.MAX_RETRIES + 1

    path = rd.save_rejected(result, tmp_path, "RUN")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["gold_ids"] == ["K001"] and len(data["attempts"]) == rd.MAX_RETRIES + 1
