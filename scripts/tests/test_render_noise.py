"""노이즈 렌더링: 문서 조립·검증·배치 재시도 (LLM 호출 없음)."""

import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import plan_noise as pn
import render_noise as rn
from common.schema.documents import Guide, SlackThread, Ticket
from common.schema.gold import load_gold_knowledge
from common.schema.knowledge import SourceType
from common.schema.world import load_world

WORLD = load_world(rn.WORLD_FILE)
GOLDS = load_gold_knowledge(rn.GOLD_DIR)
GOLD_BY_ID = {g.id: g for g in GOLDS}
USERS = rn.load_users(rn.USERS_FILE)
USERS_BY_ID = {u.user_id: u for u in USERS}
SLACK_TO_USER = {u.slack_user_id: u.user_id for u in USERS}
PLAN = pn.plan_noise(GOLDS, WORLD, USERS)


def _item(**match: Any) -> pn.NoisePlanItem:
    return next(i for i in PLAN if all(getattr(i, k) == v for k, v in match.items()))


def _ctx(item: pn.NoisePlanItem, no: int = 1) -> rn.NoiseContext:
    return rn.NoiseContext.build(item, USERS_BY_ID, GOLD_BY_ID, no)


def _mention(ctx: rn.NoiseContext) -> str:
    """원문이 반드시 담아야 할 이름·코드."""
    it = ctx.item
    parts = [it.system, it.error_code or "", it.permission or ""]
    return " ".join(p for p in parts if p)


def _slack_draft(ctx: rn.NoiseContext, texts: list[tuple[str, str]]) -> rn.SlackDraft:
    """texts: (보낸 사람 'r'|'a', 내용)."""
    who = {"r": ctx.requester, "a": ctx.assignee}
    start = ctx.item.created_at
    return rn.SlackDraft(
        title="문의",
        channel="#it-helpdesk",
        messages=[
            {
                "slack_user_id": who[w].slack_user_id,
                "ts": start + timedelta(minutes=7 * k),
                "text": t,
            }
            for k, (w, t) in enumerate(texts)
        ],
    )


def _good_slack(ctx: rn.NoiseContext) -> rn.SlackDraft:
    return _slack_draft(
        ctx,
        [("r", f"{_mention(ctx)} 관련 문의드려요"), ("a", "넵 안내드릴게요"), ("r", "감사함다")],
    )


def _validate(ctx: rn.NoiseContext, draft: Any) -> tuple[Any, list[str]]:
    doc = rn.build_noise_document(ctx, draft, 900, SLACK_TO_USER)
    return doc, rn.validate_noise_document(doc, ctx, WORLD, set(USERS_BY_ID), GOLDS)


def _ticket(ctx: rn.NoiseContext, text: str | None = None) -> dict[str, Any]:
    t0 = ctx.item.created_at
    return {
        "no": ctx.no,
        "title": "요청",
        "description": text or f"{_mention(ctx)} 처리 요청드립니다.",
        "comments": [
            {
                "user_id": ctx.assignee.user_id,
                "created_at": (t0 + timedelta(minutes=30)).isoformat(),
                "text": "확인 후 처리했습니다.",
            }
        ],
        "resolution": "처리 완료",
        "resolved_at": (t0 + timedelta(hours=1)).isoformat(),
    }


# ---------------------------------------------------------------------------
# 문서 조립·검증
# ---------------------------------------------------------------------------


def test_slack_general_passes_and_carries_plan_labels():
    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", error_code="E-103"))
    doc, errors = _validate(ctx, _good_slack(ctx))
    assert errors == []
    assert isinstance(doc, SlackThread)
    assert doc.gold_ids == [] and doc.related_gold_ids == []
    assert (doc.plan_id, doc.noise_kind, doc.issue_category) == (ctx.plan_id, "general", "VPN")
    assert doc.created_at == ctx.item.created_at and doc.doc_id == "SL-0900"


def test_gold_only_error_code_and_meta_rejected():
    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", error_code="E-103"))
    draft = _slack_draft(
        ctx,
        [("r", "VPN E-103 떠요"), ("a", "E-102랑 비슷한 건가요? plan 확인"), ("r", "NS-0003 참고")],
    )
    _, errors = _validate(ctx, draft)
    assert any("[error_code] 계획에 없는 오류 코드 E-102" in e for e in errors)
    assert any(e.startswith("[meta]") and "plan" in e for e in errors)
    assert any(e.startswith("[meta]") and "계획 ID" in e for e in errors)


def test_plan_code_must_appear_and_first_message_from_requester():
    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", error_code="E-103"))
    draft = _slack_draft(ctx, [("a", "VPN 공지입니다"), ("r", "넵"), ("a", "감사합니다")])
    _, errors = _validate(ctx, draft)
    assert any("계획 오류 코드 E-103" in e for e in errors)
    assert any("첫 메시지는 요청자" in e for e in errors)


def test_general_doc_must_not_mention_gold_policy_permission():
    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", system="형상관리"))
    group = ctx.requester.user_group
    pairs = pn.gold_policy_pairs(GOLDS)
    perm = next(p for g, p in sorted(pairs) if g == group)
    draft = _slack_draft(ctx, [("r", f"형상관리 {perm} 도 필요해요"), ("a", "넵"), ("r", "ㅇㅋ")])
    _, errors = _validate(ctx, draft)
    assert any(e.startswith("[policy]") and perm in e for e in errors)


def test_unresolved_modes():
    no_answer = _ctx(_item(source_type=SourceType.SLACK, unresolved_mode="no_answer"))
    texts = [("r", f"{_mention(no_answer)} 문의요"), ("r", "추가로"), ("r", "혹시 아시는 분?")]
    assert _validate(no_answer, _slack_draft(no_answer, texts))[1] == []
    _, errors = _validate(no_answer, _good_slack(no_answer))
    assert any(e.startswith("[unresolved]") for e in errors)

    will = _ctx(_item(source_type=SourceType.SLACK, unresolved_mode="will_check"))
    ok = [("r", f"{_mention(will)} 문의요"), ("a", "어떤 화면인가요?"), ("a", "확인해 보겠습니다!")]
    assert _validate(will, _slack_draft(will, ok))[1] == []
    _, errors = _validate(will, _good_slack(will))
    assert any("확인해보겠습니다" in e for e in errors)


def test_repeat_rejected_ticket_keeps_plan_status():
    item = _item(source_type=SourceType.TICKET, noise_kind="repeat", status="REJECTED")
    ctx = _ctx(item)
    doc, errors = _validate(ctx, rn.NoiseTicketDraft.model_validate(_ticket(ctx)))
    assert errors == []
    assert isinstance(doc, Ticket)
    assert doc.status == "REJECTED" and doc.category == "권한신청"
    assert doc.related_gold_ids == item.related_gold_ids
    assert doc.requester_user_id == item.requester_user_id


def test_repeat_allows_only_its_gold_code():
    item = _item(source_type=SourceType.SLACK, noise_kind="repeat", error_code="E-102")
    ctx = _ctx(item)
    assert ctx.allowed_codes() == {"E-102"}
    assert _validate(ctx, _good_slack(ctx))[1] == []


def test_dates_outside_window_rejected():
    ctx = _ctx(_item(source_type=SourceType.TICKET, noise_kind="general"))
    data = _ticket(ctx)
    data["resolved_at"] = (ctx.item.created_at + timedelta(days=10)).isoformat()
    _, errors = _validate(ctx, rn.NoiseTicketDraft.model_validate(data))
    assert any(e.startswith("[date]") for e in errors)


def test_guide_document():
    ctx = _ctx(_item(source_type=SourceType.GUIDE, system="메일"))
    body = (
        "## 개요\n메일함 용량이 차면 E-501이 표시됩니다.\n\n## 정리 방법\n오래된 메일을 지웁니다."
    )
    doc, errors = _validate(ctx, rn.GuideDraft(title="메일함 정리", version="1.0", body=body))
    assert errors == []
    assert isinstance(doc, Guide) and doc.updated_at == ctx.item.created_at.date()
    _, errors = _validate(ctx, rn.GuideDraft(title="t", version="1.0", body="## 개요\n메일"))
    assert any("[format]" in e for e in errors) and any("E-501" in e for e in errors)


def test_prompt_contains_plan_and_hides_labels():
    item = _item(source_type=SourceType.SLACK, noise_kind="repeat", status="REJECTED")
    msgs = rn.build_messages([_ctx(item)], WORLD, GOLDS)
    user = msgs[1]["content"]
    assert item.permission in user and f"{item.created_at:%Y-%m-%dT%H:%M:%S}" in user
    assert item.plan_id not in user and item.related_gold_ids[0] not in user
    general = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", system="VPN"))
    assert "다루지 말 것" in rn.build_messages([general], WORLD, GOLDS)[1]["content"]


# ---------------------------------------------------------------------------
# 배치·재시도·이어하기
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


def _ticket_batch() -> list[rn.NoiseContext]:
    tickets = [i for i in PLAN if i.source_type is SourceType.TICKET][:5]
    return [_ctx(it, no) for no, it in enumerate(tickets, 1)]


def test_ticket_batch_retries_only_failed_items():
    ctxs = _ticket_batch()
    first = [_ticket(c) for c in ctxs]
    first[2]["description"] = f"{_mention(ctxs[2])} E-999 오류"  # 계획에 없는 코드
    del first[4]  # 5번 누락
    retry = [_ticket(ctxs[2]), _ticket(ctxs[4])]
    client = FakeClient([json.dumps({"tickets": first}), json.dumps({"tickets": retry})])
    logged: list[rn.Attempt] = []
    result = rn.render_batch(
        client, "dep", ctxs, WORLD, USERS_BY_ID, GOLDS, log_call=lambda _u, a: logged.append(a)
    )
    assert [c.plan_id for c, _ in result.accepted] == [c.plan_id for c in ctxs]
    assert result.rejected == []
    assert [len(a.plan_ids) for a in logged] == [5, 2]
    assert set(logged[0].failed) == {ctxs[2].plan_id, ctxs[4].plan_id}
    retry_prompt = client.calls[1]["messages"][-1]["content"]
    assert (
        "[상황 3]" in retry_prompt and "[상황 5]" in retry_prompt and "[상황 1]" not in retry_prompt
    )
    assert client.calls[0]["reasoning_effort"] == "low"


def test_slack_rejected_after_max_retries():
    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", error_code="E-103"))
    bad = _slack_draft(ctx, [("r", "VPN 문의"), ("a", "E-102 같네요"), ("r", "넵")])
    client = FakeClient([bad.model_dump_json()] * (rn.MAX_RETRIES + 1))
    result = rn.render_batch(
        client, "dep", [ctx], WORLD, USERS_BY_ID, GOLDS, log_call=lambda *_: None
    )
    assert result.accepted == [] and len(result.attempts) == rn.MAX_RETRIES + 1
    assert result.rejected[0][0].plan_id == ctx.plan_id


def test_pending_items_and_batches(tmp_path):
    done = {PLAN[0].plan_id}
    pend = rn.pending_items(PLAN, done)
    assert PLAN[0].plan_id not in {i.plan_id for i in pend}
    tickets = rn.pending_items(PLAN, set(), SourceType.TICKET, limit=12)
    assert [len(b) for b in rn.make_batches(tickets)] == [5, 5, 2]
    slack = rn.pending_items(PLAN, set(), SourceType.SLACK, limit=3)
    assert [len(b) for b in rn.make_batches(slack)] == [1, 1, 1]

    ctx = _ctx(_item(source_type=SourceType.SLACK, noise_kind="general", error_code="E-103"))
    doc = rn.build_noise_document(ctx, _good_slack(ctx), 31, SLACK_TO_USER)
    rn.append_document(doc, tmp_path)
    assert rn.done_plan_ids(tmp_path) == {ctx.plan_id}


@pytest.mark.parametrize("st", list(rn.RESPONSE_MODELS))
def test_response_schema_is_strict(st):
    schema = rn.strict_json_schema(rn.RESPONSE_MODELS[st])
    assert schema["additionalProperties"] is False


def test_noise_report_counts_targets_distribution_and_quality():
    import docs_report as dr

    docs = {
        "slack": [
            {"source_type": "slack", "gold_ids": ["K001"]},
            {"source_type": "slack", "noise_kind": "unresolved", "issue_category": "VPN"},
        ],
        "ticket": [
            {
                "source_type": "ticket",
                "noise_kind": "repeat",
                "issue_category": "권한 신청",
                "status": "REJECTED",
            },
        ],
        "guide": [],
        "minutes": [],
    }
    usage = [
        {
            "attempt": 0,
            "plan_ids": ["NT-0001", "NT-0002"],
            "ok_plan_ids": ["NT-0001"],
            "failed": {"NT-0002": ["[meta] x"]},
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
        {
            "attempt": 1,
            "plan_ids": ["NT-0002"],
            "ok_plan_ids": ["NT-0002"],
            "failed": {},
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    ]
    text = dr.build_noise_report(docs, usage, 0, {"slack": 270})
    assert "slack      2 / 300  (gold 1, 노이즈 1, 노이즈 계획 270)" in text
    assert "미해결 1" in text and "REJECTED 1" in text and "권한 신청 1" in text
    assert "1차 통과율 (항목 단위) 1/2 (50%)" in text and "재시도 호출 1회" in text
    assert "meta 1" in text
    # gold 리포트는 노이즈를 세지 않는다
    assert "slack      1" in dr.build_report([], docs, [], 0)
