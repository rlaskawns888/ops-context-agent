"""노이즈 계획 → 원문 렌더링.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/datagen/plan_noise.py                         # 계획 먼저
    uv run python scripts/datagen/render_noise.py --limit 1 --dry-run
    uv run python scripts/datagen/render_noise.py --source-type ticket --limit 10
    uv run python scripts/datagen/render_noise.py   # 전체 (생성된 plan_id는 건너뜀)

입력: data/generated/noise_plan.jsonl, gold·world·users
출력: data/generated/{slack,ticket,guide}.jsonl 끝에 추가 (doc_id는 기존 번호에 이어서)
거부: data/generated/_rejected/noise-*.json
호출 로그: data/generated/noise_usage.jsonl (docs_report.py가 합산)

호출 단위: slack 1건, ticket 5건(BATCH_SIZE), guide 1건. 계획의 시스템·오류 코드·권한·인물·날짜·
status는 코드가 문서에 그대로 넣고, LLM은 제목·본문·댓글·해결 내용만 쓴다.

검증 (저장 전, validate_noise_document) — 실패한 항목만 사유를 붙여 최대 MAX_RETRIES회 재요청
- [plan] 계획 필드와 문서 메타 필드 일치 (plan_id·noise_kind·분류·인물·작성 시각·status 등),
         slack 첫 메시지는 요청자가 계획 시각에, 등장인물은 요청자·담당자만
- [unresolved] no_answer: 요청자 메시지만 / will_check: 담당자의 "확인해보겠습니다"로 끝남
- [meta] gold·Kxxx·plan·noise·계획 ID(NS-0001 등) 등 생성 메타 정보 노출 금지
- [user] 등장 user_id가 users.jsonl에 존재
- [name] 계획 시스템(guide는 정식 명칭)·권한이 원문에 언급됨
- [error_code] 원문의 E-xxx는 계획 코드(repeat은 그 gold 코드)만, 계획 코드는 반드시 등장
- [policy] general·unresolved(·guide)에 gold 정책 조합 (요청자 그룹, 권한) 언급 금지
- [date] 문서 날짜가 작성일 ~ 작성일+DOC_SPAN_DAYS (repeat은 gold 유효 범위 안)
- [format] 문서 모델 검증, guide는 ## 섹션 2개 이상
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError

from common.schema.documents import (
    Guide,
    SlackThread,
    Ticket,
    TicketCategory,
    TicketComment,
    format_doc_id,
)
from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.knowledge import EntityType, NonBlankStr, SourceType
from common.schema.users import MockUser
from common.schema.world import World, load_world
from generate_gold import PriceSettings, Usage, append_usage, format_cost
from gold_rules import DATE_MAX
from plan_noise import (
    LOCK_CATEGORY,
    PLAN_FILE,
    NoisePlanItem,
    gold_policy_pairs,
    gold_role_at,
    load_plan,
)
from render_docs import (
    GENERATED_DIR,
    GOLD_DIR,
    MAX_RETRIES,
    REASONING_EFFORT,
    REJECTED_DIR,
    USERS_FILE,
    WORLD_FILE,
    GuideDraft,
    Part,
    Role,
    SlackDraft,
    _Draft,
    _facts,
    _short_errors,
    append_document,
    check_meta,
    date_window,
    existing_state,
    filtered_knowledge,
    load_users,
    output_file,
    read_documents,
    referenced_user_ids,
    required_evidence,
    strict_json_schema,
)

USAGE_LOG = GENERATED_DIR / "noise_usage.jsonl"
REJECTED_PREFIX = "noise-"
NOISE_SOURCE_TYPES: tuple[SourceType, ...] = (SourceType.SLACK, SourceType.TICKET, SourceType.GUIDE)
BATCH_SIZE: dict[SourceType, int] = {SourceType.SLACK: 1, SourceType.TICKET: 5, SourceType.GUIDE: 1}
MAX_COMPLETION_TOKENS: dict[SourceType, int] = {
    SourceType.SLACK: 8000,
    SourceType.TICKET: 20000,
    SourceType.GUIDE: 8000,
}
DOC_SPAN_DAYS = 3  # 문서 안 날짜(답변·댓글·해결)는 작성일부터 이 기간 안

Doc = SlackThread | Ticket | Guide

# ---------------------------------------------------------------------------
# 항목 맥락
# ---------------------------------------------------------------------------


@dataclass
class NoiseContext:
    """계획 항목 1개 + 렌더링에 필요한 고정 값."""

    item: NoisePlanItem
    requester: MockUser | None
    assignee: MockUser
    gold: GoldKnowledge | None  # repeat이면 다시 묻는 gold
    role: Role
    window: tuple[date, date]  # 문서 속 날짜 허용 범위
    no: int = 1  # 배치 안 번호 (프롬프트 [상황 n])

    @property
    def plan_id(self) -> str:
        return self.item.plan_id

    @property
    def source_type(self) -> SourceType:
        return self.item.source_type

    @classmethod
    def build(
        cls,
        item: NoisePlanItem,
        users: dict[str, MockUser],
        golds: dict[str, GoldKnowledge],
        no: int = 1,
    ) -> NoiseContext:
        day = item.created_at.date()
        gold = golds[item.related_gold_ids[0]] if item.related_gold_ids else None
        role: Role = gold_role_at(gold, day) if gold else "full"
        end = min(day + timedelta(days=DOC_SPAN_DAYS), DATE_MAX)
        if gold is not None:
            end = min(end, date_window(gold, role)[1])
        if item.source_type is SourceType.GUIDE:
            end = day
        requester = users[item.requester_user_id] if item.requester_user_id else None
        return cls(item, requester, users[item.assignee_user_id], gold, role, (day, end), no)

    @property
    def cast(self) -> list[MockUser]:
        return [u for u in (self.requester, self.assignee) if u is not None]

    def ticket_category(self) -> TicketCategory:
        it = self.item
        if it.permission:
            return "권한신청"
        if it.category == LOCK_CATEGORY:
            return "계정"
        has_error = it.error_code is not None or (
            self.gold is not None and any(e.type is EntityType.ERROR for e in self.gold.entities)
        )
        return "장애" if has_error else "문의"

    def allowed_codes(self) -> set[str]:
        codes = {self.item.error_code} if self.item.error_code else set()
        if self.gold is not None:
            codes |= {e.code for e in self.gold.entities if e.type is EntityType.ERROR and e.code}
        return codes


# ---------------------------------------------------------------------------
# 구조화 출력 모델 (LLM이 채우는 부분만)
# ---------------------------------------------------------------------------


class NoiseTicketDraft(_Draft):
    no: int = Field(description="[상황 n]의 n")
    title: NonBlankStr
    description: NonBlankStr
    comments: list[TicketComment]
    resolution: NonBlankStr
    resolved_at: datetime


class NoiseTicketBatch(_Draft):
    tickets: list[NoiseTicketDraft]


RESPONSE_MODELS: dict[SourceType, type[_Draft]] = {
    SourceType.SLACK: SlackDraft,
    SourceType.TICKET: NoiseTicketBatch,
    SourceType.GUIDE: GuideDraft,
}


# ---------------------------------------------------------------------------
# draft → 문서
# ---------------------------------------------------------------------------


def build_noise_document(
    ctx: NoiseContext, draft: _Draft, doc_number: int, slack_to_user: dict[str, str]
) -> Doc:
    """draft에 계획 값(ID·인물·날짜·status·라벨)을 붙여 문서로 만든다. 틀리면 ValidationError."""
    it = ctx.item
    st = it.source_type
    common = {
        "doc_id": format_doc_id(st, doc_number),
        "source_type": st,
        "gold_ids": [],
        "related_gold_ids": list(it.related_gold_ids),
        "issue_category": it.category,
        "noise_kind": it.noise_kind,
        "plan_id": it.plan_id,
        "title": draft.title,  # type: ignore[attr-defined]
    }
    if isinstance(draft, SlackDraft):
        authors = list(
            dict.fromkeys(
                slack_to_user.get(m.slack_user_id, m.slack_user_id) for m in draft.messages
            )
        )
        created = draft.messages[0].ts if draft.messages else it.created_at
        return SlackThread.model_validate(
            {
                **common,
                "created_at": created,
                "author_user_ids": authors,
                "channel": draft.channel,
                "messages": [m.model_dump() for m in draft.messages],
            }
        )
    if isinstance(draft, NoiseTicketDraft):
        assert ctx.requester is not None
        authors = list(
            dict.fromkeys(
                [ctx.requester.user_id, ctx.assignee.user_id, *(c.user_id for c in draft.comments)]
            )
        )
        return Ticket.model_validate(
            {
                **common,
                "created_at": it.created_at,
                "author_user_ids": authors,
                "ticket_no": f"SR-{it.created_at:%Y%m%d}-{doc_number:04d}",
                "category": ctx.ticket_category(),
                "status": it.status,
                "requester_user_id": ctx.requester.user_id,
                "assignee_user_id": ctx.assignee.user_id,
                "resolved_at": draft.resolved_at,
                "description": draft.description,
                "comments": [c.model_dump() for c in draft.comments],
                "resolution": draft.resolution,
            }
        )
    if isinstance(draft, GuideDraft):
        return Guide.model_validate(
            {
                **common,
                "created_at": it.created_at,
                "author_user_ids": [ctx.assignee.user_id],
                "version": draft.version,
                "updated_at": it.created_at.date(),
                "body": draft.body,
            }
        )
    raise TypeError(f"알 수 없는 draft 타입: {type(draft).__name__}")


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

NOISE_META_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"plan", re.IGNORECASE), "plan"),
    (re.compile(r"noise", re.IGNORECASE), "noise"),
    (re.compile(r"노이즈"), "노이즈"),
    (re.compile(r"(?<![A-Za-z0-9])N[STGM]-\d{3,4}(?!\d)"), "계획 ID"),
    (
        re.compile(
            r"related_gold|issue_category|noise_kind|unresolved_mode|requester_user_id",
            re.IGNORECASE,
        ),
        "생성 필드명",
    ),
]
_ERROR_CODE = re.compile(r"E-\d{3}")
_GUIDE_SECTION = re.compile(r"^##\s+\S", re.MULTILINE)
WILL_CHECK_PHRASE = "확인해보겠습니다"


def check_noise_meta(texts: Iterable[str]) -> list[str]:
    texts = list(texts)
    found = check_meta(texts)
    for text in texts:
        for pattern, label in NOISE_META_PATTERNS:
            m = pattern.search(text)
            if m and f"{label}: {m.group(0)!r}" not in found:
                found.append(f"{label}: {m.group(0)!r}")
    return found


def _names(world: World, kind: str, canonical: str) -> list[str]:
    items = {"system": world.systems, "permission": world.permissions}[kind]
    for i in items:
        if i.canonical_name == canonical:
            return [i.canonical_name, *i.aliases]
    return [canonical]


def forbidden_permissions(ctx: NoiseContext, golds: Sequence[GoldKnowledge]) -> list[str]:
    """general·unresolved·guide 원문이 언급하면 안 되는 권한 (gold 정책 조합)."""
    if ctx.item.noise_kind == "repeat":
        return []
    pairs = gold_policy_pairs(golds)
    if ctx.requester is None:  # guide: 어느 그룹이든 gold 정책이 있는 권한은 다루지 않는다
        perms = {p for _, p in pairs}
    else:
        perms = {p for g, p in pairs if g == ctx.requester.user_group}
    return sorted(perms - {ctx.item.permission})


def _is_will_check(text: str) -> bool:
    return WILL_CHECK_PHRASE in "".join(text.split())


def validate_noise_document(
    doc: Doc,
    ctx: NoiseContext,
    world: World,
    user_ids: set[str],
    golds: Sequence[GoldKnowledge],
) -> list[str]:
    """저장 전 검증. 위반 사유 목록 (비어 있으면 통과). 사유 앞에 [분류]를 붙인다."""
    it = ctx.item
    errors: list[str] = []
    texts = list(doc.text_fields())
    joined = "\n".join(texts)

    # -- 계획 필드 일치
    expect = {
        "plan_id": it.plan_id,
        "noise_kind": it.noise_kind,
        "issue_category": it.category,
        "related_gold_ids": it.related_gold_ids,
        "gold_ids": [],
        "source_type": it.source_type,
        "created_at": it.created_at,
    }
    if isinstance(doc, Ticket):
        expect |= {
            "requester_user_id": it.requester_user_id,
            "assignee_user_id": it.assignee_user_id,
            "status": it.status,
        }
    if isinstance(doc, Guide):
        expect |= {"author_user_ids": [it.assignee_user_id], "updated_at": it.created_at.date()}
    for name, value in expect.items():
        if getattr(doc, name) != value:
            errors.append(f"[plan] {name}={getattr(doc, name)!r} (계획 {value!r})")

    cast_ids = {u.user_id for u in ctx.cast}
    if isinstance(doc, SlackThread):
        if doc.created_at != it.created_at:
            errors.append(f"[plan] 첫 메시지 시각은 {it.created_at:%Y-%m-%dT%H:%M:%S}이어야 함")
        first = doc.author_user_ids[0] if doc.author_user_ids else None
        if first != it.requester_user_id:
            errors.append("[plan] 첫 메시지는 요청자가 써야 함")
        senders = [_slack_sender(m.slack_user_id, ctx) for m in doc.messages]
        if it.unresolved_mode == "no_answer":
            if any(s != it.requester_user_id for s in senders):
                errors.append("[unresolved] 답변 없음 스레드는 요청자 메시지만 있어야 함")
        else:
            if it.assignee_user_id not in senders:
                errors.append("[plan] 담당자 메시지가 없음")
        if it.unresolved_mode == "will_check" and not (
            senders[-1] == it.assignee_user_id and _is_will_check(doc.messages[-1].text)
        ):
            errors.append(f"[unresolved] 마지막 메시지는 담당자의 '{WILL_CHECK_PHRASE}'여야 함")
    if (
        isinstance(doc, Ticket)
        and doc.comments
        and doc.resolved_at < max(c.created_at for c in doc.comments)
    ):
        errors.append("[format] resolved_at이 마지막 댓글보다 빠름")
    others = [uid for uid in referenced_user_ids(doc) if uid not in cast_ids]
    if others:
        errors.append(f"[plan] 등장인물은 요청자·담당자만: {', '.join(others)}")

    # -- 메타·사용자
    errors += [f"[meta] 메타 정보 노출 — {m}" for m in check_noise_meta(texts)]
    for uid in referenced_user_ids(doc):
        if uid not in user_ids:
            errors.append(f"[user] 등장인물 ID {uid!r}가 users.jsonl에 없음")

    # -- 이름
    if isinstance(doc, Guide):
        if it.system not in joined:
            errors.append(f"[name] 가이드에 시스템 정식 명칭 {it.system!r}이 없음")
    elif not any(n in joined for n in _names(world, "system", it.system)):
        errors.append(f"[name] 시스템 {it.system!r}(또는 별칭)이 원문에 언급되지 않음")
    if it.permission and not any(n in joined for n in _names(world, "permission", it.permission)):
        errors.append(f"[name] 권한 {it.permission!r}(또는 별칭)이 원문에 언급되지 않음")

    # -- 오류 코드
    allowed = ctx.allowed_codes()
    found = set(_ERROR_CODE.findall(joined))
    for code in sorted(found - allowed):
        errors.append(f"[error_code] 계획에 없는 오류 코드 {code}")
    if it.error_code and it.error_code not in found:
        errors.append(f"[error_code] 계획 오류 코드 {it.error_code}가 원문에 없음")

    # -- gold 정책 조합
    for perm in forbidden_permissions(ctx, golds):
        hit = next((n for n in _names(world, "permission", perm) if n in joined), None)
        if hit:
            errors.append(f"[policy] 다루지 않아야 할 권한 {perm!r}(표기 {hit!r}) 언급")

    # -- 날짜
    start, end = ctx.window
    bad = sorted({d for d in doc.dates() if not start <= d <= end})
    if bad:
        errors.append(f"[date] 날짜 {', '.join(map(str, bad))}가 허용 범위 {start} ~ {end} 밖")

    if isinstance(doc, Guide) and len(_GUIDE_SECTION.findall(doc.body)) < 2:
        errors.append("[format] guide body는 ## 섹션이 2개 이상이어야 함")
    return errors


def _slack_sender(slack_user_id: str, ctx: NoiseContext) -> str:
    for u in ctx.cast:
        if u.slack_user_id == slack_user_id:
            return u.user_id
    return slack_user_id


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

_SYSTEM_RULES = """\
너는 사내 IT 운영 환경의 합성 원문 작가다. 주어진 [상황]대로 실제 사내 문서처럼 보이는 원문을
JSON으로 작성한다.

[절대 규칙]
1. [상황]의 시스템·오류 코드·권한·등장인물·날짜·처리 결과를 바꾸지 않는다.
2. 오류 코드(E-로 시작하는 코드)는 [상황]에 적힌 코드만 쓴다. [상황]에 오류 코드가 없으면
   어떤 오류 코드도 쓰지 않는다.
3. 생성 메타 정보를 원문에 쓰지 않는다: 상황 번호, N이나 K로 시작하는 번호, gold, plan, noise,
   노이즈, 분류 이름(일반·재문의·미해결 등), 영어 필드명.
4. 사람은 [등장인물]만 등장시키고, ID 필드(slack_user_id, user_id)에는 주어진 ID를 그대로 쓴다.
   본문에서는 이름으로 부른다.
5. 시스템·팀·권한·사용자 그룹은 [이름 사전]의 canonical_name 또는 aliases로만 부른다.
6. [다루지 말 것]에 적힌 주제·권한·오류는 언급하지 않는다.
7. 시각 형식 YYYY-MM-DDTHH:MM:SS.
"""

# 문체·aliases 규칙은 render_docs._STYLE과 같다 (필수 문장 관련 문구만 뺐다)
_STYLE: dict[SourceType, str] = {
    SourceType.SLACK: """\
[원문 유형: Slack 스레드]
- 사내 메신저 구어체. 가벼운 오타·줄임말(ㅠㅠ, 넵, ㅇㅋ, 감사함다 등)을 약간 섞는다.
- 시스템·권한 이름은 절반 정도 aliases로 자연스럽게 부른다.
- 메시지 3~10개. 질문자가 먼저 묻는다. ts는 시간순, 같은 날 몇 분~몇 시간 간격.
- channel은 #으로 시작하는 영문 소문자·하이픈 채널명 (예: #it-helpdesk).""",
    SourceType.TICKET: """\
[원문 유형: IT 요청 티켓 — 한 번에 여러 건]
- 업무 문체. 시스템·권한 이름은 일부 aliases를 섞는다.
- [상황 n]마다 티켓 1건을 tickets 배열에 쓰고, no에는 n을 넣는다.
- description: 요청자가 쓴 요청·장애 내용. comments: 담당자·요청자의 처리 기록 1~4개(시간순,
  접수 시각 이후). resolution: 담당자의 처리 결과 요약. resolved_at: 마지막 댓글 이후 시각.
- comments의 user_id에는 U로 시작하는 user_id를 쓴다 (slack_user_id가 아니다).
- 처리 결과(해결·반려)는 [상황]에 적힌 대로 쓴다.
- 티켓끼리 제목·문장·표현이 겹치지 않게 다양하게 쓴다.""",
    SourceType.GUIDE: """\
[원문 유형: 사내 공식 가이드 문서]
- 공식 문체(~합니다). 시스템은 canonical_name으로 부른다.
- body는 markdown. "## 제목"으로 2개 이상 섹션을 나눈다 (예: 개요, 적용 대상, 절차, 문의처).
  body에 "# 문서 제목"은 쓰지 않는다 (title 필드에 쓴다).
- version은 "1.0", "2.1" 같은 형식.""",
}


def _outcome(ctx: NoiseContext) -> str:
    it = ctx.item
    if it.source_type is SourceType.GUIDE:
        return "일반 안내 문서. 특정 사용자 그룹에 대한 권한 허용·금지 정책은 쓰지 않는다."
    if it.unresolved_mode == "no_answer":
        return (
            "미해결 — 아무도 답하지 않는다. 메시지는 모두 요청자가 쓴다(질문, 추가 설명, "
            "'혹시 아시는 분 계실까요?' 같은 재문의). 해결되지 않은 채 끝난다."
        )
    if it.unresolved_mode == "will_check":
        return (
            "미해결 — 담당자는 상황만 묻고 해결 방법은 쓰지 않는다. 마지막 메시지는 담당자가 "
            f"'{WILL_CHECK_PHRASE}'라는 말로 끝맺는다."
        )
    if it.noise_kind == "repeat":
        result = "반려된다 (정책상 부여할 수 없음)" if it.status == "REJECTED" else "해결된다"
        return (
            f"{result}. 예전에도 있었던 같은 문제를 이 요청자가 다시 묻는 상황이다. 해결 방법·정책 "
            "판단은 [기존 지식]과 일치해야 하고, 문장은 그대로 베끼지 말고 자기 말로 쓴다."
        )
    if it.permission:
        return (
            "해결된다 — 일반 신청 절차(소속 팀장 확인 등)를 거쳐 권한이 부여된다. 사용자 그룹별 "
            "허용·금지 정책은 언급하지 않는다."
        )
    return "해결된다 — 담당자가 일반적인 IT 지원 수준의 방법으로 안내해 해결된다."


def _person_line(u: MockUser, label: str) -> str:
    who = f"{u.it_team} 담당자" if u.it_team else f"{u.department} / {u.user_group}"
    return f"  - {label}: {u.name} (user_id={u.user_id}, slack_user_id={u.slack_user_id}, {who})"


def _date_line(ctx: NoiseContext) -> str:
    it, (_, end) = ctx.item, ctx.window
    ts = f"{it.created_at:%Y-%m-%dT%H:%M:%S}"
    if it.source_type is SourceType.SLACK:
        return f"첫 메시지 ts는 정확히 {ts}. 이후 메시지는 시간순, {end} 이전"
    if it.source_type is SourceType.TICKET:
        return f"접수 시각 {ts}. 댓글·resolved_at은 접수 이후 {end} 23:59 이전"
    return f"개정일 {it.created_at.date()}"


def situation_block(ctx: NoiseContext, world: World, golds: Sequence[GoldKnowledge]) -> str:
    it = ctx.item
    lines = [f"[상황 {ctx.no}]" if it.source_type is SourceType.TICKET else "[상황]"]
    lines.append(f"- 분류: {it.category}")
    lines.append(f"- 시스템: {it.system}")
    if it.error_code:
        e = next(e for e in world.error_codes if e.code == it.error_code)
        lines.append(
            f'- 오류 코드: {e.code} ("{e.message}" — {e.description}). '
            f"원문(요청 내용·메시지·댓글 중 하나)에 '{e.code}'를 그대로 한 번 이상 쓴다"
        )
    else:
        lines.append("- 오류 코드: 없음 (오류 코드를 쓰지 않는다)")
    if it.permission:
        group = ctx.requester.user_group if ctx.requester else "-"
        lines.append(f"- 신청 권한: {it.permission} (요청자 소속 그룹: {group})")
    lines.append(f"- 주제: {it.topic}")
    lines.append(f"- 처리 결과: {_outcome(ctx)}")
    lines.append(f"- 날짜: {_date_line(ctx)}")
    lines.append("- 등장인물:")
    if ctx.requester is not None:
        label = "질문자" if it.source_type is SourceType.SLACK else "요청자"
        lines.append(_person_line(ctx.requester, label))
    label = "작성 부서 담당자(본문에 이름을 쓸 필요는 없음)" if ctx.requester is None else "담당자"
    lines.append(_person_line(ctx.assignee, label))

    if ctx.gold is not None:
        part = Part(ctx.gold, ctx.role)
        facts = yaml.safe_dump(_facts(part), allow_unicode=True, sort_keys=False, width=200)
        refs = "\n".join(f"  - {s}" for s in required_evidence(ctx.gold, ctx.role))
        lines.append("- 기존 지식 (해결 방법·정책 판단의 근거):\n" + _indent(facts))
        lines.append("- 기존 지식 요약 문장 (뜻은 지키되 그대로 베끼지 않는다):\n" + refs)
    else:
        avoid = _avoid_lines(ctx, golds)
        if avoid:
            lines.append("- 다루지 말 것:\n" + "\n".join(f"  - {a}" for a in avoid))
    return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.rstrip().splitlines())


def _avoid_lines(ctx: NoiseContext, golds: Sequence[GoldKnowledge]) -> list[str]:
    """general·unresolved·guide: 같은 시스템의 gold 주제와 gold 정책 권한은 다루지 않는다."""
    out = [
        g.title.split(" (")[0]
        for g in golds
        if ctx.item.system in {e.canonical_name for e in g.entities if e.type is EntityType.SYSTEM}
    ]
    perms = forbidden_permissions(ctx, golds)
    if perms:
        out.append("다음 권한은 언급하지 않는다: " + ", ".join(perms))
    return out


def name_dictionary(ctxs: Sequence[NoiseContext], world: World) -> dict[str, list[dict]]:
    systems = {c.item.system for c in ctxs}
    teams = {s.owner_team for s in world.systems if s.canonical_name in systems}
    teams |= {c.assignee.it_team for c in ctxs if c.assignee.it_team}
    groups = {c.requester.user_group for c in ctxs if c.requester}
    perms = {c.item.permission for c in ctxs if c.item.permission}
    for c in ctxs:
        if c.gold is not None:
            ents, _, pols = filtered_knowledge(c.gold, c.role)
            systems |= {e.canonical_name for e in ents if e.type is EntityType.SYSTEM}
            teams |= {e.canonical_name for e in ents if e.type is EntityType.TEAM}
            groups |= {p.subject_group for p in pols}
            perms |= {e.canonical_name for e in ents if e.type is EntityType.PERMISSION}
    out: dict[str, list[dict]] = {}
    for kind, items, names in (
        ("systems", world.systems, systems),
        ("teams", world.teams, teams),
        ("user_groups", world.user_groups, groups),
        ("permissions", world.permissions, perms),
    ):
        picked = [
            {"canonical_name": i.canonical_name, "aliases": i.aliases}
            for i in items
            if i.canonical_name in names
        ]
        if picked:
            out[kind] = picked
    return out


def build_messages(
    ctxs: Sequence[NoiseContext], world: World, golds: Sequence[GoldKnowledge]
) -> list[dict[str, str]]:
    st = ctxs[0].source_type
    system = "\n\n".join([_SYSTEM_RULES, _STYLE[st]])
    names = yaml.safe_dump(name_dictionary(ctxs, world), allow_unicode=True, sort_keys=False)
    user = "\n\n".join([*(situation_block(c, world, golds) for c in ctxs), "[이름 사전]\n" + names])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    attempt: int
    plan_ids: list[str]  # 이번 호출이 다룬 항목
    failed: dict[str, list[str]]  # plan_id → 실패 사유 (통과 항목은 없음)

    @property
    def ok_plan_ids(self) -> list[str]:
        return [p for p in self.plan_ids if p not in self.failed]


@dataclass
class BatchResult:
    accepted: list[tuple[NoiseContext, _Draft]]
    rejected: list[tuple[NoiseContext, list[str]]]
    attempts: list[Attempt]
    last_content: str


def _drafts_by_ctx(
    response: Any, ctxs: Sequence[NoiseContext]
) -> tuple[dict[str, _Draft], dict[str, list[str]], str]:
    """응답 → (plan_id별 draft, plan_id별 실패 사유, 원본 content)."""
    choice = response.choices[0]
    content = choice.message.content or ""
    fail_all = None
    if getattr(choice.message, "refusal", None):
        fail_all = f"[schema] 모델 거부: {choice.message.refusal}"
    elif choice.finish_reason == "length":
        fail_all = "[schema] 출력이 max_completion_tokens에서 잘림. 더 간결하게."
    if fail_all:
        return {}, {c.plan_id: [fail_all] for c in ctxs}, content
    st = ctxs[0].source_type
    try:
        parsed = RESPONSE_MODELS[st].model_validate_json(content)
    except ValidationError as exc:
        return {}, {c.plan_id: [f"[schema] {_short_errors(exc)}"] for c in ctxs}, content
    if isinstance(parsed, NoiseTicketBatch):
        by_no: dict[int, NoiseTicketDraft] = {}
        for t in parsed.tickets:
            by_no.setdefault(t.no, t)
        drafts: dict[str, _Draft] = {}
        missing: dict[str, list[str]] = {}
        for c in ctxs:
            if c.no in by_no:
                drafts[c.plan_id] = by_no[c.no]
            else:
                missing[c.plan_id] = [f"[schema] no={c.no} 티켓이 응답에 없음"]
        return drafts, missing, content
    return {ctxs[0].plan_id: parsed}, {}, content


def evaluate_response(
    response: Any,
    ctxs: Sequence[NoiseContext],
    world: World,
    users: dict[str, MockUser],
    golds: Sequence[GoldKnowledge],
) -> tuple[dict[str, _Draft], dict[str, list[str]], str]:
    """응답 1개 → (통과한 plan_id별 draft, 실패한 plan_id별 사유, 원본 content)."""
    drafts, failed, content = _drafts_by_ctx(response, ctxs)
    slack_to_user = {u.slack_user_id: u.user_id for u in users.values()}
    ok: dict[str, _Draft] = {}
    for c in ctxs:
        draft = drafts.get(c.plan_id)
        if draft is None:
            continue
        try:
            doc = build_noise_document(c, draft, 1, slack_to_user)
        except ValidationError as exc:
            failed[c.plan_id] = [f"[format] {_short_errors(exc)}"]
            continue
        reasons = validate_noise_document(doc, c, world, set(users), golds)
        if reasons:
            failed[c.plan_id] = reasons
        else:
            ok[c.plan_id] = draft
    return ok, failed, content


def _retry_message(ctxs: Sequence[NoiseContext], failed: dict[str, list[str]]) -> str:
    if ctxs[0].source_type is not SourceType.TICKET:
        reasons = failed[ctxs[0].plan_id]
        return "검증 실패. 아래 사유를 모두 고쳐 전체 JSON을 다시 출력하라.\n" + "\n".join(
            f"- {r}" for r in reasons
        )
    lines = [
        "검증 실패. 아래 상황의 티켓만 사유를 고쳐 다시 작성하라. "
        "tickets 배열에는 다시 쓰는 티켓만 넣고 no는 그대로 둔다."
    ]
    for c in ctxs:
        if c.plan_id in failed:
            lines.append(f"[상황 {c.no}]")
            lines += [f"- {r}" for r in failed[c.plan_id]]
    return "\n".join(lines)


def render_batch(
    client: Any,
    deployment: str,
    ctxs: Sequence[NoiseContext],
    world: World,
    users: dict[str, MockUser],
    golds: Sequence[GoldKnowledge],
    *,
    log_call: Callable[[Usage, Attempt], None],
) -> BatchResult:
    """배치 1개 생성. 실패한 항목만 사유를 붙여 최대 MAX_RETRIES회 재요청."""
    st = ctxs[0].source_type
    messages = build_messages(ctxs, world, golds)
    schema = strict_json_schema(RESPONSE_MODELS[st])
    pending = list(ctxs)
    accepted: list[tuple[NoiseContext, _Draft]] = []
    attempts: list[Attempt] = []
    failed: dict[str, list[str]] = {}
    content = ""
    for i in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            name=f"render_noise.{st.value}",
            model=deployment,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"noise_{st.value}_draft",
                    "strict": True,
                    "schema": schema,
                },
            },
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS[st],
        )
        ok, failed, content = evaluate_response(response, pending, world, users, golds)
        attempt = Attempt(i, [c.plan_id for c in pending], failed)
        attempts.append(attempt)
        log_call(Usage.from_response(response), attempt)
        accepted += [(c, ok[c.plan_id]) for c in pending if c.plan_id in ok]
        pending = [c for c in pending if c.plan_id in failed]
        if not pending:
            break
        if i < MAX_RETRIES:
            print(f"  [retry {i + 1}/{MAX_RETRIES}] {len(pending)}건 실패")
            for c in pending:
                for r in failed[c.plan_id][:3]:
                    print(f"      - {c.plan_id}: {r[:200]}")
        messages = [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": _retry_message(pending, failed)},
        ]
    order = {c.plan_id: k for k, c in enumerate(ctxs)}
    accepted.sort(key=lambda pair: order[pair[0].plan_id])
    rejected = [(c, failed[c.plan_id]) for c in pending]
    return BatchResult(accepted, rejected, attempts, content)


# ---------------------------------------------------------------------------
# 이어하기·저장
# ---------------------------------------------------------------------------


def done_plan_ids(out_dir: Path = GENERATED_DIR) -> set[str]:
    return {
        d["plan_id"]
        for st in NOISE_SOURCE_TYPES
        for d in read_documents(output_file(st, out_dir))
        if d.get("plan_id")
    }


def pending_items(
    plan: Sequence[NoisePlanItem],
    done: set[str],
    source_type: SourceType | None = None,
    limit: int | None = None,
) -> list[NoisePlanItem]:
    out = [
        i
        for i in plan
        if i.plan_id not in done and (source_type is None or i.source_type is source_type)
    ]
    return out if limit is None else out[:limit]


def make_batches(items: Sequence[NoisePlanItem]) -> list[list[NoisePlanItem]]:
    """source_type별로 BATCH_SIZE씩 (계획 순서 유지)."""
    batches: list[list[NoisePlanItem]] = []
    for st in NOISE_SOURCE_TYPES:
        sub = [i for i in items if i.source_type is st]
        size = BATCH_SIZE[st]
        batches += [sub[k : k + size] for k in range(0, len(sub), size)]
    return batches


def save_rejected(
    ctx: NoiseContext,
    reasons_by_attempt: list[list[str]],
    content: str,
    rejected_dir: Path,
    run_id: str,
) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    path = rejected_dir / f"{REJECTED_PREFIX}{run_id}-{ctx.plan_id}.json"
    data = {
        "plan_id": ctx.plan_id,
        "source_type": ctx.source_type.value,
        "plan": ctx.item.model_dump(mode="json"),
        "attempts": [{"attempt": k, "reasons": r} for k, r in enumerate(reasons_by_attempt)],
        "last_content": content,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="노이즈 계획 → 원문(slack·ticket·guide) 렌더링")
    parser.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 문서 수")
    parser.add_argument(
        "--source-type", choices=[s.value for s in NOISE_SOURCE_TYPES], help="이 유형만 생성"
    )
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    parser.add_argument("--plan", type=Path, default=PLAN_FILE, help="계획 파일")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상")
    return args


@dataclass
class RunTotals:
    usage: Usage = field(default_factory=Usage)
    costs: list[float | None] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0


def main(argv: Sequence[str] | None = None) -> int:
    from generate_gold import _make_client

    args = parse_args(argv)
    if not args.plan.exists():
        raise SystemExit(f"{args.plan}가 없습니다. plan_noise.py를 먼저 실행하세요.")
    world = load_world(WORLD_FILE)
    golds = load_gold_knowledge(GOLD_DIR)
    golds_by_id = {g.id: g for g in golds}
    users = load_users(USERS_FILE)
    users_by_id = {u.user_id: u for u in users}
    slack_to_user = {u.slack_user_id: u.user_id for u in users}
    _, next_number = existing_state(GENERATED_DIR)
    source_type = SourceType(args.source_type) if args.source_type else None
    items = pending_items(load_plan(args.plan), done_plan_ids(), source_type, args.limit)
    batches = make_batches(items)
    print(
        f"대상 노이즈 문서 {len(items)}개, 호출 단위 {len(batches)}개"
        f"{' (dry-run)' if args.dry_run else ''}"
    )
    if not items:
        return 0

    prices = PriceSettings()
    client, langfuse, deployment = _make_client()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    totals = RunTotals()
    started = time.monotonic()

    def make_logger(st: SourceType) -> Callable[[Usage, Attempt], None]:
        def log_call(usage: Usage, attempt: Attempt) -> None:
            cost = usage.cost_usd(prices)
            totals.costs.append(cost)
            for f in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
                setattr(totals.usage, f, getattr(totals.usage, f) + getattr(usage, f))
            print(
                f"  [call] {st.value} {','.join(attempt.plan_ids)} attempt={attempt.attempt} "
                f"in={usage.prompt_tokens} (cached {usage.cached_tokens}) "
                f"out={usage.completion_tokens} (reasoning {usage.reasoning_tokens}) "
                f"cost={format_cost(cost)} ok={len(attempt.ok_plan_ids)}/{len(attempt.plan_ids)}"
            )
            append_usage(
                USAGE_LOG,
                {
                    "run_id": run_id,
                    "ts": datetime.now(UTC).isoformat(),
                    "source_type": st.value,
                    "plan_ids": attempt.plan_ids,
                    "ok_plan_ids": attempt.ok_plan_ids,
                    "failed": attempt.failed,
                    "model": deployment,
                    "attempt": attempt.attempt,
                    "dry_run": args.dry_run,
                    **usage.__dict__,
                    "cost_usd": cost,
                },
            )

        return log_call

    try:
        for batch in batches:
            st = batch[0].source_type
            ctxs = [
                NoiseContext.build(it, users_by_id, golds_by_id, no)
                for no, it in enumerate(batch, 1)
            ]
            result = render_batch(
                client, deployment, ctxs, world, users_by_id, golds, log_call=make_logger(st)
            )
            for ctx, draft in result.accepted:
                number = next_number[st]
                doc = build_noise_document(ctx, draft, number, slack_to_user)
                next_number[st] = number + 1
                totals.accepted += 1
                print(f"  [OK] {doc.doc_id} ← {ctx.plan_id}")
                if args.dry_run:
                    print(doc.model_dump_json(indent=2))
                else:
                    append_document(doc, GENERATED_DIR)
            for ctx, reasons in result.rejected:
                totals.rejected += 1
                print(f"  [REJECT] {ctx.plan_id}")
                for r in reasons:
                    print(f"      - {r[:300]}")
                if not args.dry_run:
                    history = [a.failed.get(ctx.plan_id, []) for a in result.attempts]
                    save_rejected(ctx, history, result.last_content, REJECTED_DIR, run_id)
    finally:
        langfuse.flush()

    costs = totals.costs
    cost = None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
    u = totals.usage
    print(
        f"\n통과 {totals.accepted} / 거부 {totals.rejected} | 호출 {len(costs)}회 | "
        f"토큰 in={u.prompt_tokens} (cached {u.cached_tokens}) "
        f"out={u.completion_tokens} (reasoning {u.reasoning_tokens}) | "
        f"비용 {format_cost(cost)} | {time.monotonic() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
