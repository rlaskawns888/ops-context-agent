"""gold → 원문 렌더링.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/datagen/render_docs.py --limit 1 --dry-run
    uv run python scripts/datagen/render_docs.py --source-type slack
    uv run python scripts/datagen/render_docs.py               # 전체 (이미 만든 것은 건너뜀)

입력: data/gold/knowledge/*.yaml, data/world/world.yaml, data/generated/users.jsonl
출력: data/generated/{slack,ticket,guide,minutes}.jsonl (한 줄 = 문서 1개, 생성 즉시 append)
거부: data/generated/_rejected/*.json
호출 로그: data/generated/docs_usage.jsonl (docs_report.py가 합산)

작업 단위 (Job)
- (gold, render_as) 1쌍 = 원문 1개.
- minutes만 예외: minutes gold를 회의록 MINUTES_DOC_COUNT개에 배치한다 (plan_minutes).
  남는 만큼 gold 2개를 한 회의록에 묶고, 같은 시스템(없으면 같은 담당 팀) gold끼리만 묶는다.
- 재실행 시 출력 파일에 이미 있는 (gold_id, source_type)은 건너뛴다 (중단 후 이어하기).

policy_revision (이전 정책 → 신규 정책)
- render_as가 2개 이상이면 원문을 '이전 정책 시점'과 '신규 정책 시점'으로 나눈다.
  guide가 있으면 guide가 신규, 나머지가 이전 (현재 데이터: minutes=이전, guide=신규).
  guide가 없으면 render_as 마지막 유형이 신규다.
- 이전 원문: 공통 사실(UserGroup·System·Permission·ON) + 이전 정책 evidence만 담고,
  날짜는 [이전 valid_from, 신규 valid_from 전날]. 신규 정책 문장이 나오면 거부.
- 신규 원문: 공통 사실 + 신규 정책 evidence(SUPERSEDES 포함), 날짜는 [신규 valid_from, valid_to].
- render_as가 1개뿐이면 그 원문이 모든 evidence를 담고 신규 원문 날짜 규칙을 따른다.
- 그 외 gold: valid_from이 있으면 그 이후(valid_to가 있으면 그 이전), 없으면 전체 범위에서 분산.

등장인물은 users.jsonl에서 고정 seed로 고른다 (pick_cast).
- 질문자: requester. gold에 정책이 있으면 정책 대상 UserGroup 사용자.
- 답변자·담당자: 해당 시스템 담당 팀(gold OWNED_BY, 없으면 world owner_team)의 it_staff.

검증 (저장 전, validate_document) — 실패 사유를 붙여 최대 MAX_RETRIES회 재요청
- [evidence] 필수 evidence 문장이 원문 필드 하나에 그대로 포함 (공백·줄바꿈 정규화 후 비교)
- [meta] gold·K0xx·evidence·scenario 등 생성 메타 정보 노출 금지
- [user] 등장 user_id·slack_user_id가 users.jsonl에 존재
- [name] gold의 System이 world canonical 또는 aliases로 언급됨 (guide는 canonical 필수)
- [error_code] 원문의 E-xxx 코드는 해당 gold의 Error 코드만
- [date] 문서 날짜가 날짜 규칙 범위 안, 이전 정책 원문에 신규 정책 문장 금지
- [format] 문서 모델 검증 (메시지 수·시간순 등), guide는 ## 섹션 2개 이상
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.schema.documents import (
    ActionItem,
    Guide,
    Minutes,
    SlackMessage,
    SlackThread,
    Ticket,
    TicketCategory,
    TicketComment,
    TicketStatus,
    format_doc_id,
)
from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.knowledge import EntityType, NonBlankStr, RelationType, SourceType
from common.schema.knowledge import _StrictJsonSchema as StrictJsonSchema
from common.schema.users import MockUser
from common.schema.world import World, load_world
from generate_gold import PriceSettings, Usage, append_usage, format_cost
from gold_rules import DATE_MAX, DATE_MIN

ROOT = Path(__file__).resolve().parents[2]
WORLD_FILE = ROOT / "data" / "world" / "world.yaml"
GOLD_DIR = ROOT / "data" / "gold" / "knowledge"
GENERATED_DIR = ROOT / "data" / "generated"
USERS_FILE = GENERATED_DIR / "users.jsonl"
REJECTED_DIR = GENERATED_DIR / "_rejected"
USAGE_LOG = GENERATED_DIR / "docs_usage.jsonl"

SOURCE_TYPES: tuple[SourceType, ...] = (
    SourceType.SLACK,
    SourceType.TICKET,
    SourceType.GUIDE,
    SourceType.MINUTES,
)
MINUTES_DOC_COUNT = 20
MAX_RETRIES = 2
MAX_COMPLETION_TOKENS = 12000
REASONING_EFFORT = "low"
DEFAULT_SEED = 42

Role = Literal["full", "old", "new"]


def output_file(source_type: SourceType, out_dir: Path = GENERATED_DIR) -> Path:
    return out_dir / f"{source_type.value}.jsonl"


# ---------------------------------------------------------------------------
# gold → 원문 몫 (Part): 필수 evidence·날짜 범위
# ---------------------------------------------------------------------------


def revision_policy_names(gold: GoldKnowledge) -> tuple[str, str]:
    """policy_revision gold의 (이전 정책, 신규 정책) 이름. SUPERSEDES 관계로 찾는다."""
    for r in gold.relations:
        if r.relation_type is RelationType.SUPERSEDES:
            return r.target_name, r.source_name
    raise ValueError(f"{gold.id}: policy_revision인데 SUPERSEDES 관계가 없습니다")


def revision_role(gold: GoldKnowledge, source_type: SourceType) -> Role:
    if gold.scenario_type != "policy_revision" or len(gold.render_as) < 2:
        return "full"
    new_type = SourceType.GUIDE if SourceType.GUIDE in gold.render_as else gold.render_as[-1]
    return "new" if source_type == new_type else "old"


def _excluded_policy(gold: GoldKnowledge, role: Role) -> str | None:
    """role에서 빠지는 정책 이름 (old면 신규 정책, new면 이전 정책)."""
    if role == "full":
        return None
    old, new = revision_policy_names(gold)
    return new if role == "old" else old


def filtered_knowledge(gold: GoldKnowledge, role: Role) -> tuple[list, list, list]:
    """role에 해당하는 (entities, relations, policies)."""
    skip = _excluded_policy(gold, role)
    entities = [
        e for e in gold.entities if not (e.type is EntityType.POLICY and e.canonical_name == skip)
    ]
    relations = [
        r
        for r in gold.relations
        if not (r.source_type is EntityType.POLICY and r.source_name == skip)
    ]
    policies = [p for p in gold.policies if p.name != skip]
    return entities, relations, policies


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def required_evidence(gold: GoldKnowledge, role: Role = "full") -> list[str]:
    """원문에 그대로 들어가야 할 evidence 문장 (중복 제거, gold 순서)."""
    entities, relations, policies = filtered_knowledge(gold, role)
    return _unique(x.evidence for x in [*entities, *relations, *policies])


def forbidden_evidence(gold: GoldKnowledge, role: Role) -> list[str]:
    """이전 정책 원문에 나오면 안 되는 문장 (신규 정책에만 있는 evidence)."""
    if role != "old":
        return []
    old = set(required_evidence(gold, "old"))
    return [s for s in required_evidence(gold, "new") if s not in old]


def date_window(gold: GoldKnowledge, role: Role = "full") -> tuple[date, date]:
    """이 gold를 담은 원문이 가질 수 있는 날짜 범위 [start, end] (양 끝 포함)."""
    start, end = DATE_MIN, DATE_MAX
    if gold.scenario_type == "policy_revision":
        old_name, new_name = revision_policy_names(gold)
        by_name = {p.name: p for p in gold.policies}
        old, new = by_name[old_name], by_name[new_name]
        if new.valid_from is None:
            raise ValueError(f"{gold.id}: 신규 정책 valid_from이 없습니다")
        if role == "old":
            start = max(start, old.valid_from or DATE_MIN)
            end = min(end, new.valid_from - timedelta(days=1))
        else:
            start = max(start, new.valid_from)
            end = min(end, new.valid_to or DATE_MAX)
        return start, end
    periods = [(p.valid_from, p.valid_to) for p in gold.policies]
    periods += [
        (e.valid_from, e.valid_to)
        for e in gold.entities
        if e.type in (EntityType.SOLUTION, EntityType.POLICY)
    ]
    for valid_from, valid_to in periods:
        if valid_from is not None:
            start = max(start, valid_from)
        if valid_to is not None:
            end = min(end, valid_to)
    return start, end


@dataclass(frozen=True)
class Part:
    """원문 1개가 담는 gold 1건과 그 몫."""

    gold: GoldKnowledge
    role: Role


def intersect_windows(windows: Iterable[tuple[date, date]]) -> tuple[date, date]:
    windows = list(windows)
    return max(w[0] for w in windows), min(w[1] for w in windows)


# ---------------------------------------------------------------------------
# 작업 계획 (Job)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    source_type: SourceType
    parts: tuple[Part, ...]

    @property
    def gold_ids(self) -> list[str]:
        return [p.gold.id for p in self.parts]

    @property
    def key(self) -> str:
        return f"{self.source_type.value}:{'+'.join(self.gold_ids)}"

    @property
    def window(self) -> tuple[date, date]:
        return intersect_windows(date_window(p.gold, p.role) for p in self.parts)

    def required_evidence(self) -> list[str]:
        return _unique(s for p in self.parts for s in required_evidence(p.gold, p.role))

    def forbidden_evidence(self) -> list[str]:
        required = set(self.required_evidence())
        return _unique(
            s for p in self.parts for s in forbidden_evidence(p.gold, p.role) if s not in required
        )


def gold_systems(gold: GoldKnowledge) -> set[str]:
    return {e.canonical_name for e in gold.entities if e.type is EntityType.SYSTEM}


def primary_systems(gold: GoldKnowledge) -> list[str]:
    """담당 팀을 정할 기준 시스템: 정책 대상 → 오류 발생 시스템 → System 엔티티 전체."""
    names = _unique(p.system_name for p in gold.policies)
    if not names:
        names = _unique(
            r.target_name for r in gold.relations if r.relation_type is RelationType.OCCURS_IN
        )
    if not names:
        names = sorted(gold_systems(gold))
    return names


def responsible_teams(gold: GoldKnowledge, world: World) -> list[str]:
    """답변·담당 팀: gold의 OWNED_BY 대상, 없으면 기준 시스템의 world owner_team."""
    owned = _unique(
        r.target_name for r in gold.relations if r.relation_type is RelationType.OWNED_BY
    )
    if owned:
        return owned
    owners = {s.canonical_name: s.owner_team for s in world.systems}
    return _unique(owners[s] for s in primary_systems(gold) if s in owners)


def plan_minutes(
    golds: Sequence[GoldKnowledge], world: World, doc_count: int = MINUTES_DOC_COUNT
) -> list[list[GoldKnowledge]]:
    """minutes gold를 회의록 doc_count개에 배치한다. 회의록 1개 = gold 1~2개.

    묶음 후보는 날짜 범위가 겹치는 쌍만. 같은 시스템을 공유하는 쌍을 먼저,
    다음으로 담당 팀이 같은 쌍을 gold 번호 순으로 골라 서로 겹치지 않게 채운다.
    """
    items = [g for g in golds if SourceType.MINUTES in g.render_as]
    pairs_needed = len(items) - doc_count
    if pairs_needed < 0 or pairs_needed > len(items) // 2:
        raise ValueError(f"minutes gold {len(items)}건을 회의록 {doc_count}개에 배치할 수 없습니다")

    def window(g: GoldKnowledge) -> tuple[date, date]:
        return date_window(g, revision_role(g, SourceType.MINUTES))

    candidates: list[tuple[int, int, int]] = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            start, end = intersect_windows([window(a), window(b)])
            if start > end:
                continue
            if gold_systems(a) & gold_systems(b):
                score = 2
            elif set(responsible_teams(a, world)) & set(responsible_teams(b, world)):
                score = 1
            else:
                continue
            candidates.append((-score, i, j))

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if len(pairs) == pairs_needed:
            break
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used |= {i, j}
    if len(pairs) < pairs_needed:
        raise ValueError(
            f"같은 시스템·팀 gold 쌍이 {len(pairs)}개뿐이라 회의록 {doc_count}개로 묶을 수 없습니다"
        )

    groups = [[items[i], items[j]] for i, j in pairs]
    groups += [[g] for k, g in enumerate(items) if k not in used]
    return sorted(groups, key=lambda grp: grp[0].id)


def plan_jobs(golds: Sequence[GoldKnowledge], world: World) -> list[Job]:
    """전체 작업 목록 (gold 번호 순, 회의록은 묶음의 첫 gold 위치)."""
    minutes_by_first = {
        grp[0].id: Job(
            SourceType.MINUTES,
            tuple(Part(g, revision_role(g, SourceType.MINUTES)) for g in grp),
        )
        for grp in plan_minutes(golds, world)
    }
    jobs: list[Job] = []
    for gold in golds:
        for source_type in gold.render_as:
            if source_type is SourceType.MINUTES:
                if gold.id in minutes_by_first:
                    jobs.append(minutes_by_first[gold.id])
                continue
            jobs.append(Job(source_type, (Part(gold, revision_role(gold, source_type)),)))
    return jobs


# ---------------------------------------------------------------------------
# 등장인물·날짜
# ---------------------------------------------------------------------------


@dataclass
class Cast:
    requesters: list[MockUser]
    staff: list[MockUser]

    @property
    def everyone(self) -> list[MockUser]:
        return [*self.requesters, *self.staff]


def _sample(rng: random.Random, pool: Sequence[MockUser], k: int) -> list[MockUser]:
    return rng.sample(list(pool), min(k, len(pool)))


def pick_cast(job: Job, users: Sequence[MockUser], world: World, rng: random.Random) -> Cast:
    groups = _unique(
        p.subject_group for part in job.parts for p in filtered_knowledge(part.gold, part.role)[2]
    )
    requesters = [u for u in users if u.role == "requester"]
    if groups:
        requesters = [u for u in requesters if u.user_group in groups] or requesters
    teams = _unique(t for part in job.parts for t in responsible_teams(part.gold, world))
    all_staff = [u for u in users if u.role == "it_staff"]
    staff_pool = [u for u in all_staff if u.it_team in teams] or all_staff

    st = job.source_type
    if st is SourceType.SLACK:
        return Cast(_sample(rng, requesters, 1), _sample(rng, staff_pool, rng.randint(1, 2)))
    if st is SourceType.TICKET:
        return Cast(_sample(rng, requesters, 1), _sample(rng, staff_pool, 1))
    if st is SourceType.GUIDE:
        return Cast([], _sample(rng, staff_pool, 1))
    # minutes: 담당 팀 운영자 2~3명 (+정책 안건이면 대상 그룹 현업 대표 1명)
    staff = _sample(rng, staff_pool, 3)
    if len(staff) < 2:
        others = [u for u in all_staff if u not in staff]
        staff += _sample(rng, others, 2 - len(staff))
    return Cast(_sample(rng, requesters, 1) if groups else [], staff)


def pick_date(window: tuple[date, date], rng: random.Random, margin_days: int = 3) -> date:
    """window 안의 평일 하나 (끝에서 margin_days만큼 여유를 둔다: 해결 시각 등이 넘치지 않게)."""
    start, end = window
    if start > end:
        raise ValueError(f"날짜 범위가 비어 있습니다: {start} ~ {end}")
    last = max(start, end - timedelta(days=margin_days))
    span = (last - start).days
    day = start
    for _ in range(20):
        day = start + timedelta(days=rng.randint(0, span))
        if day.weekday() < 5:
            break
    return day


def job_rng(seed: int, job: Job) -> random.Random:
    return random.Random(f"{seed}:{job.key}")  # str seed는 PYTHONHASHSEED와 무관하게 결정적


@dataclass
class JobContext:
    """프롬프트·문서 조립에 쓰는 고정 값 (job마다 seed로 결정)."""

    job: Job
    cast: Cast
    doc_date: date
    created_at: datetime  # ticket·minutes·guide는 이 값을 그대로 쓴다

    @classmethod
    def build(
        cls, job: Job, users: Sequence[MockUser], world: World, seed: int = DEFAULT_SEED
    ) -> JobContext:
        rng = job_rng(seed, job)
        cast = pick_cast(job, users, world, rng)
        doc_date = pick_date(job.window, rng)
        hour = rng.randint(9, 17) if job.source_type is not SourceType.GUIDE else 9
        created_at = datetime.combine(doc_date, dtime(hour, rng.choice([0, 10, 20, 30, 40, 50])))
        return cls(job, cast, doc_date, created_at)


# ---------------------------------------------------------------------------
# 구조화 출력 모델 (LLM이 채우는 부분만. ID·날짜·등장인물 배정은 코드가 정한다)
# ---------------------------------------------------------------------------


class _Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlackDraft(_Draft):
    title: NonBlankStr = Field(description="스레드 내용을 요약한 짧은 제목")
    channel: NonBlankStr = Field(description="#으로 시작하는 채널명")
    messages: list[SlackMessage]


class TicketDraft(_Draft):
    title: NonBlankStr
    category: TicketCategory
    status: TicketStatus
    description: NonBlankStr
    comments: list[TicketComment]
    resolution: NonBlankStr
    resolved_at: datetime


class GuideDraft(_Draft):
    title: NonBlankStr
    version: NonBlankStr
    body: NonBlankStr


class MinutesDraft(_Draft):
    title: NonBlankStr
    agenda: list[NonBlankStr]
    discussion: NonBlankStr
    decisions: list[NonBlankStr]
    action_items: list[ActionItem]


DRAFT_MODELS: dict[SourceType, type[_Draft]] = {
    SourceType.SLACK: SlackDraft,
    SourceType.TICKET: TicketDraft,
    SourceType.GUIDE: GuideDraft,
    SourceType.MINUTES: MinutesDraft,
}

_UNSUPPORTED_KEYWORDS = frozenset({"format", "default", "minLength", "maxLength", "pattern"})


def response_json_schema(source_type: SourceType) -> dict[str, Any]:
    """source_type별 draft 모델의 strict JSON 스키마 (Azure OpenAI response_format용)."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in _UNSUPPORTED_KEYWORDS or k == "properties"
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    model = DRAFT_MODELS[source_type]
    return strip(model.model_json_schema(schema_generator=StrictJsonSchema))


# ---------------------------------------------------------------------------
# draft → 문서
# ---------------------------------------------------------------------------


def build_document(
    ctx: JobContext,
    draft: _Draft,
    doc_number: int,
    slack_to_user: dict[str, str],
) -> SlackThread | Ticket | Guide | Minutes:
    """draft에 코드가 정한 값(ID·날짜·등장인물)을 붙여 문서 모델로 만든다.

    틀리면 ValidationError.
    """
    job = ctx.job
    st = job.source_type
    common = {
        "doc_id": format_doc_id(st, doc_number),
        "source_type": st,
        "gold_ids": job.gold_ids,
        "title": draft.title,
    }
    if isinstance(draft, SlackDraft):
        authors = _unique(
            slack_to_user.get(m.slack_user_id, m.slack_user_id) for m in draft.messages
        )
        created = draft.messages[0].ts if draft.messages else ctx.created_at
        return SlackThread.model_validate(
            {
                **common,
                "created_at": created,
                "author_user_ids": authors,
                "channel": draft.channel,
                "messages": [m.model_dump() for m in draft.messages],
            }
        )
    if isinstance(draft, TicketDraft):
        requester, assignee = ctx.cast.requesters[0], ctx.cast.staff[0]
        authors = _unique(
            [requester.user_id, assignee.user_id, *(c.user_id for c in draft.comments)]
        )
        return Ticket.model_validate(
            {
                **common,
                "created_at": ctx.created_at,
                "author_user_ids": authors,
                "ticket_no": f"SR-{ctx.created_at:%Y%m%d}-{doc_number:04d}",
                "requester_user_id": requester.user_id,
                "assignee_user_id": assignee.user_id,
                **draft.model_dump(exclude={"title"}),
            }
        )
    if isinstance(draft, GuideDraft):
        return Guide.model_validate(
            {
                **common,
                "created_at": ctx.created_at,
                "author_user_ids": [u.user_id for u in ctx.cast.staff],
                "version": draft.version,
                "updated_at": ctx.doc_date,
                "body": draft.body,
            }
        )
    if isinstance(draft, MinutesDraft):
        attendees = [u.user_id for u in ctx.cast.everyone]
        return Minutes.model_validate(
            {
                **common,
                "created_at": ctx.created_at,
                "author_user_ids": attendees,
                "meeting_date": ctx.doc_date,
                "attendees": attendees,
                **draft.model_dump(exclude={"title"}),
            }
        )
    raise TypeError(f"알 수 없는 draft 타입: {type(draft).__name__}")


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

META_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gold", re.IGNORECASE), "gold"),
    (re.compile(r"(?<![A-Za-z0-9])K\d{3}(?!\d)"), "지식 ID(Kxxx)"),
    (re.compile(r"evidence", re.IGNORECASE), "evidence"),
    (re.compile(r"scenario", re.IGNORECASE), "scenario"),
    (
        re.compile(
            r"render_as|canonical_name|knowledge_type|tacit|official_procedure"
            r"|policy_revision|multihop",
            re.IGNORECASE,
        ),
        "생성 필드명·유형명",
    ),
    (re.compile(r"필수\s*문장|정답\s*데이터|암묵지"), "생성 메타 용어"),
]
_ERROR_CODE = re.compile(r"E-\d{3}")
_GUIDE_SECTION = re.compile(r"^##\s+\S", re.MULTILINE)


def normalize_text(text: str) -> str:
    """공백·줄바꿈을 공백 1개로 모은다 (evidence 부분 문자열 비교용)."""
    return " ".join(text.split())


def check_evidence(texts: Sequence[str], sentences: Iterable[str]) -> list[str]:
    """sentences 중 어느 필드에도 그대로 들어가지 않은 문장 목록."""
    normalized = [normalize_text(t) for t in texts]
    return [s for s in sentences if not any(normalize_text(s) in t for t in normalized)]


def check_meta(texts: Iterable[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for pattern, label in META_PATTERNS:
            m = pattern.search(text)
            if m and f"{label}: {m.group(0)!r}" not in found:
                found.append(f"{label}: {m.group(0)!r}")
    return found


def referenced_user_ids(doc: SlackThread | Ticket | Guide | Minutes) -> list[str]:
    ids = list(doc.author_user_ids)
    if isinstance(doc, Ticket):
        ids += [doc.requester_user_id, doc.assignee_user_id, *(c.user_id for c in doc.comments)]
    elif isinstance(doc, Minutes):
        ids += [*doc.attendees, *(a.owner_user_id for a in doc.action_items)]
    return _unique(ids)


def validate_document(
    doc: SlackThread | Ticket | Guide | Minutes,
    job: Job,
    world: World,
    user_ids: set[str],
) -> list[str]:
    """저장 전 검증. 위반 사유 목록 (비어 있으면 통과). 사유 앞에 [분류]를 붙인다."""
    errors: list[str] = []
    texts = list(doc.text_fields())
    joined = "\n".join(texts)

    errors += [
        f"[evidence] 다음 문장이 원문에 그대로 없음: {s}"
        for s in check_evidence(texts, job.required_evidence())
    ]
    errors += [f"[meta] 메타 정보 노출 — {m}" for m in check_meta(texts)]

    for uid in referenced_user_ids(doc):
        if uid not in user_ids:
            errors.append(f"[user] 등장인물 ID {uid!r}가 users.jsonl에 없음")

    systems = {s.canonical_name: s for s in world.systems}
    for part in job.parts:
        for name in sorted(gold_systems(part.gold)):
            wdef = systems.get(name)
            if wdef is None:
                errors.append(f"[name] 시스템 {name!r}가 world에 없음")
                continue
            if doc.source_type is SourceType.GUIDE:
                if name not in joined:
                    errors.append(f"[name] 가이드에 시스템 정식 명칭 {name!r}이 없음")
            elif not any(n in joined for n in [name, *wdef.aliases]):
                errors.append(f"[name] 시스템 {name!r}(또는 별칭)이 원문에 언급되지 않음")

    allowed_codes = {
        e.code for part in job.parts for e in part.gold.entities if e.type is EntityType.ERROR
    }
    for code in sorted(set(_ERROR_CODE.findall(joined)) - allowed_codes):
        errors.append(f"[error_code] 주어진 사실에 없는 오류 코드 {code}")

    start, end = job.window
    bad_dates = sorted({d for d in doc.dates() if not start <= d <= end})
    if bad_dates:
        errors.append(
            f"[date] 날짜 {', '.join(map(str, bad_dates))}가 허용 범위 {start} ~ {end} 밖"
        )
    forbidden = job.forbidden_evidence()
    absent = set(check_evidence(texts, forbidden))
    for s in forbidden:
        if s not in absent:
            errors.append(f"[date] 개정 전 시점 문서에 신규 정책 문장이 있음: {s}")

    if isinstance(doc, Guide) and len(_GUIDE_SECTION.findall(doc.body)) < 2:
        errors.append("[format] guide body는 ## 섹션이 2개 이상이어야 함")
    return errors


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

_SYSTEM_RULES = """\
너는 사내 IT 운영 환경의 합성 원문 작가다. 주어진 [사실]과 [필수 문장]으로 실제 사내 문서처럼
보이는 원문 1개를 JSON으로 작성한다.

[절대 규칙]
1. [필수 문장]은 한 글자도 바꾸지 말고 그대로 원문에 넣는다. 문장 중간에 줄바꿈·마크다운 기호
   (**, `, 링크 등)를 넣지 않고, 한 문장은 한 필드(메시지 하나, 본문 하나, 항목 하나) 안에
   통째로 넣는다.
   문장 앞뒤에 다른 말을 이어 붙이는 것은 괜찮다.
2. [사실]에 없는 해결법·정책·오류·오류 코드를 새로 만들지 않는다. 인사말·감사·잡담·진행 상황은 자유.
3. 생성 메타 정보를 원문에 쓰지 않는다: gold, evidence, scenario, 지식 번호(K로 시작하는 번호),
   필수 문장, 정답, 암묵지 같은 분류 용어, 영어 필드명.
4. 사람은 [등장인물]만 등장시키고, ID 필드(slack_user_id, user_id, owner_user_id)에는 주어진 ID를
   그대로 쓴다. 본문에서는 이름으로 부른다.
5. 시스템·팀·권한·사용자 그룹은 [이름 사전]의 canonical_name 또는 aliases로만 부른다.
6. 날짜는 [날짜] 범위를 지킨다. 시각 형식 YYYY-MM-DDTHH:MM:SS, 날짜 형식 YYYY-MM-DD.
"""

_STYLE: dict[SourceType, str] = {
    SourceType.SLACK: """\
[원문 유형: Slack 스레드]
- 사내 메신저 구어체. 가벼운 오타·줄임말(ㅠㅠ, 넵, ㅇㅋ, 감사함다 등)을 약간 섞는다
  (단 [필수 문장]은 원문 그대로).
- 시스템·권한 이름은 절반 정도 aliases로 자연스럽게 부른다.
- 메시지 3~10개. 질문자가 먼저 묻고 담당자가 답한다. ts는 시간순, 같은 날 몇 분~몇 시간 간격.
- channel은 #으로 시작하는 영문 소문자·하이픈 채널명 (예: #it-helpdesk).""",
    SourceType.TICKET: """\
[원문 유형: IT 요청 티켓]
- 업무 문체. 시스템·권한 이름은 일부 aliases를 섞는다.
- description: 요청자가 쓴 요청·장애 내용. comments: 담당자·요청자의 처리 기록 1~4개(시간순,
  접수 시각 이후). resolution: 담당자의 처리 결과 요약. resolved_at: 마지막 댓글 이후 시각.
- 정책상 허용되지 않아 요청을 들어줄 수 없으면 status=REJECTED, 처리되면 RESOLVED.""",
    SourceType.GUIDE: """\
[원문 유형: 사내 공식 가이드 문서]
- 공식 문체(~합니다). 시스템은 canonical_name으로 부른다 ([필수 문장] 속 표기는 그대로 둔다).
- body는 markdown. "## 제목"으로 2개 이상 섹션을 나눈다 (예: 개요, 적용 대상, 절차, 문의처).
  body에 "# 문서 제목"은 쓰지 않는다 (title 필드에 쓴다).
- version은 "1.0", "2.1" 같은 형식.""",
    SourceType.MINUTES: """\
[원문 유형: 회의록]
- 개조식 회의록 문체. agenda: 안건 목록. discussion: 참석자 이름과 함께 논의 내용(여러 문단 가능).
  decisions: 결정 사항. action_items: 담당자(owner_user_id)·할 일·기한(due_date, 없으면 null).
- 안건이 여러 개면 각 안건의 내용을 모두 다룬다.""",
}

_ROLE_NOTE: dict[Role, str] = {
    "full": "",
    "old": (
        "이 원문은 정책 개정 **이전** 시점의 문서다. [사실]의 현행 정책만 다루고, "
        "정책이 바뀐다거나 개정·폐지된다는 내용은 쓰지 않는다."
    ),
    "new": (
        "이 원문은 정책 개정 **이후** 시점의 문서다. 신규 정책과, 그것이 이전 정책을 대체했다는 "
        "사실을 담는다."
    ),
}


def _facts(part: Part) -> dict[str, Any]:
    entities, relations, policies = filtered_knowledge(part.gold, part.role)
    return {
        # knowledge_type(tacit/official)은 원문에 노출되면 안 되는 분류 값이라 빼고 준다
        "entities": [
            e.model_dump(mode="json", exclude={"evidence", "knowledge_type"}) for e in entities
        ],
        "relations": [
            f"{r.source_type}({r.source_name}) -{r.relation_type}-> "
            f"{r.target_type}({r.target_name})"
            for r in relations
        ],
        "policies": [p.model_dump(mode="json", exclude={"evidence"}) for p in policies],
    }


def name_dictionary(job: Job, world: World) -> dict[str, list[dict[str, Any]]]:
    """job에 나오는 시스템·팀·그룹·권한의 world 이름 (canonical + aliases)."""
    names: set[str] = set()
    for part in job.parts:
        entities, _, policies = filtered_knowledge(part.gold, part.role)
        names |= {e.canonical_name for e in entities}
        names |= {p.system_name for p in policies} | {p.subject_group for p in policies}
    for team in (t for p in job.parts for t in responsible_teams(p.gold, world)):
        names.add(team)
    out: dict[str, list[dict[str, Any]]] = {}
    for kind, items in (
        ("systems", world.systems),
        ("teams", world.teams),
        ("user_groups", world.user_groups),
        ("permissions", world.permissions),
    ):
        picked = [
            {"canonical_name": i.canonical_name, "aliases": i.aliases}
            for i in items
            if i.canonical_name in names
        ]
        if picked:
            out[kind] = picked
    return out


def _person_line(u: MockUser, label: str) -> str:
    who = f"{u.it_team} 담당자" if u.it_team else f"{u.department} / {u.user_group}"
    return f"- {label}: {u.name} (user_id={u.user_id}, slack_user_id={u.slack_user_id}, {who})"


def _date_section(ctx: JobContext) -> str:
    start, end = ctx.job.window
    st = ctx.job.source_type
    lines = [f"- 허용 범위: {start} ~ {end} (원문 속 모든 날짜·시각이 이 안에 있어야 한다)"]
    if st is SourceType.SLACK:
        lines.append(f"- 스레드 날짜: {ctx.doc_date} (첫 메시지는 이 날 업무 시간)")
    elif st is SourceType.TICKET:
        lines.append(f"- 티켓 접수 시각(created_at): {ctx.created_at:%Y-%m-%dT%H:%M:%S}")
    elif st is SourceType.GUIDE:
        lines.append(f"- 문서 개정일(updated_at): {ctx.doc_date}")
    else:
        lines.append(f"- 회의 날짜: {ctx.doc_date}")
    return "\n".join(lines)


def build_messages(ctx: JobContext, world: World) -> list[dict[str, str]]:
    job = ctx.job
    st = job.source_type
    system = "\n\n".join([_SYSTEM_RULES, _STYLE[st]])

    facts = []
    for i, part in enumerate(job.parts, 1):
        head = f"## 안건 {i}" if len(job.parts) > 1 else "## 내용"
        note = _ROLE_NOTE[part.role]
        body = yaml.safe_dump(_facts(part), allow_unicode=True, sort_keys=False, width=200)
        facts.append("\n".join(x for x in [head, note, body] if x))

    cast_lines: list[str] = []
    if st is SourceType.SLACK:
        cast_lines += [_person_line(u, "질문자") for u in ctx.cast.requesters]
        cast_lines += [_person_line(u, "답변자") for u in ctx.cast.staff]
    elif st is SourceType.TICKET:
        cast_lines += [_person_line(u, "요청자") for u in ctx.cast.requesters]
        cast_lines += [_person_line(u, "담당자") for u in ctx.cast.staff]
    elif st is SourceType.GUIDE:
        cast_lines += [
            _person_line(u, "작성 부서 담당자(본문에 이름을 쓸 필요는 없음)")
            for u in ctx.cast.staff
        ]
    else:
        cast_lines += [_person_line(u, "참석자(현업 대표)") for u in ctx.cast.requesters]
        cast_lines += [_person_line(u, "참석자") for u in ctx.cast.staff]

    required = "\n".join(f"{i}. {s}" for i, s in enumerate(job.required_evidence(), 1))
    names = yaml.safe_dump(name_dictionary(job, world), allow_unicode=True, sort_keys=False)
    user = "\n\n".join(
        [
            "[사실]\n" + "\n\n".join(facts),
            "[필수 문장] — 각 문장을 원문 그대로 포함할 것\n" + required,
            "[이름 사전]\n" + names,
            "[등장인물]\n" + "\n".join(cast_lines),
            "[날짜]\n" + _date_section(ctx),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    attempt: int
    reasons: list[str]
    evidence_ok: bool | None  # 응답을 문서로 만들지 못했으면 None


@dataclass
class JobResult:
    job: Job
    document: SlackThread | Ticket | Guide | Minutes | None
    attempts: list[Attempt]
    last_content: str

    @property
    def accepted(self) -> bool:
        return self.document is not None


def evaluate_response(
    response: Any,
    ctx: JobContext,
    world: World,
    users: dict[str, MockUser],
    doc_number: int,
) -> tuple[SlackThread | Ticket | Guide | Minutes | None, list[str], bool | None, str]:
    """응답 1개 → (문서 또는 None, 실패 사유, evidence 통과 여부, 원본 content)."""
    choice = response.choices[0]
    content = choice.message.content or ""
    if getattr(choice.message, "refusal", None):
        return None, [f"[schema] 모델 거부: {choice.message.refusal}"], None, content
    if choice.finish_reason == "length":
        return None, ["[schema] 출력이 max_completion_tokens에서 잘림. 더 간결하게."], None, content
    st = ctx.job.source_type
    try:
        draft = DRAFT_MODELS[st].model_validate_json(content)
    except ValidationError as exc:
        return None, [f"[schema] {_short_errors(exc)}"], None, content
    slack_to_user = {u.slack_user_id: u.user_id for u in users.values()}
    try:
        doc = build_document(ctx, draft, doc_number, slack_to_user)
    except ValidationError as exc:
        return None, [f"[format] {_short_errors(exc)}"], None, content
    reasons = validate_document(doc, ctx.job, world, set(users))
    evidence_ok = not any(r.startswith("[evidence]") for r in reasons)
    return (None if reasons else doc), reasons, evidence_ok, content


def _short_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{err['msg']} @ {'.'.join(map(str, err['loc']))}" for err in exc.errors()[:10]
    )


def render_job(
    client: Any,
    deployment: str,
    ctx: JobContext,
    world: World,
    users: dict[str, MockUser],
    doc_number: int,
    *,
    log_call: Callable[[Usage, Attempt], None],
) -> JobResult:
    """원문 1개 생성. 검증 실패 시 사유를 붙여 최대 MAX_RETRIES회 재요청."""
    st = ctx.job.source_type
    messages = build_messages(ctx, world)
    schema = response_json_schema(st)
    attempts: list[Attempt] = []
    content = ""
    for i in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            name=f"render_docs.{st.value}",
            model=deployment,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": f"{st.value}_draft", "strict": True, "schema": schema},
            },
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        doc, reasons, evidence_ok, content = evaluate_response(
            response, ctx, world, users, doc_number
        )
        attempt = Attempt(i, reasons, evidence_ok)
        attempts.append(attempt)
        log_call(Usage.from_response(response), attempt)
        if doc is not None:
            return JobResult(ctx.job, doc, attempts, content)
        if i < MAX_RETRIES:
            print(f"  [retry {i + 1}/{MAX_RETRIES}] {ctx.job.key}: {len(reasons)}건 실패")
            for r in reasons[:5]:
                print(f"      - {r[:200]}")
        messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "검증 실패. 아래 사유를 모두 고쳐 전체 JSON을 다시 출력하라.\n"
                + "\n".join(f"- {r}" for r in reasons),
            },
        ]
    return JobResult(ctx.job, None, attempts, content)


# ---------------------------------------------------------------------------
# 저장·이어하기
# ---------------------------------------------------------------------------


def read_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def existing_state(out_dir: Path) -> tuple[set[tuple[str, str]], dict[SourceType, int]]:
    """이미 생성된 (gold_id, source_type) 집합과 유형별 다음 문서 번호."""
    done: set[tuple[str, str]] = set()
    next_number: dict[SourceType, int] = {}
    for st in SOURCE_TYPES:
        docs = read_documents(output_file(st, out_dir))
        for d in docs:
            done |= {(gid, st.value) for gid in d["gold_ids"]}
        numbers = [int(d["doc_id"].split("-")[1]) for d in docs]
        next_number[st] = max(numbers, default=0) + 1
    return done, next_number


def pending_jobs(
    jobs: Sequence[Job],
    done: set[tuple[str, str]],
    source_type: SourceType | None = None,
    limit: int | None = None,
) -> list[Job]:
    out = [
        j
        for j in jobs
        if (source_type is None or j.source_type is source_type)
        and not all((gid, j.source_type.value) in done for gid in j.gold_ids)
    ]
    return out if limit is None else out[:limit]


def append_document(doc: SlackThread | Ticket | Guide | Minutes, out_dir: Path) -> None:
    path = output_file(doc.source_type, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(doc.model_dump_json() + "\n")


def save_rejected(result: JobResult, rejected_dir: Path, run_id: str) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    name = result.job.key.replace(":", "-").replace("+", "_")
    path = rejected_dir / f"{run_id}-{name}.json"
    data = {
        "job": result.job.key,
        "source_type": result.job.source_type.value,
        "gold_ids": result.job.gold_ids,
        "attempts": [{"attempt": a.attempt, "reasons": a.reasons} for a in result.attempts],
        "last_content": result.last_content,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_users(path: Path) -> list[MockUser]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [MockUser.model_validate_json(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gold → 원문(slack·ticket·guide·minutes) 렌더링")
    parser.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 원문 수")
    parser.add_argument(
        "--source-type", choices=[s.value for s in SOURCE_TYPES], help="이 유형만 생성"
    )
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="등장인물·날짜 seed")
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
    world = load_world(WORLD_FILE)
    golds = load_gold_knowledge(GOLD_DIR)
    users = load_users(USERS_FILE)
    users_by_id = {u.user_id: u for u in users}
    done, next_number = existing_state(GENERATED_DIR)
    source_type = SourceType(args.source_type) if args.source_type else None
    jobs = pending_jobs(plan_jobs(golds, world), done, source_type, args.limit)
    print(f"대상 원문 {len(jobs)}개{' (dry-run)' if args.dry_run else ''}")
    if not jobs:
        return 0

    prices = PriceSettings()
    client, langfuse, deployment = _make_client()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    totals = RunTotals()
    started = time.monotonic()

    def make_logger(job: Job) -> Callable[[Usage, Attempt], None]:
        def log_call(usage: Usage, attempt: Attempt) -> None:
            cost = usage.cost_usd(prices)
            totals.costs.append(cost)
            for f in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
                setattr(totals.usage, f, getattr(totals.usage, f) + getattr(usage, f))
            print(
                f"  [call] {job.key} attempt={attempt.attempt} in={usage.prompt_tokens} "
                f"(cached {usage.cached_tokens}) out={usage.completion_tokens} "
                f"(reasoning {usage.reasoning_tokens}) cost={format_cost(cost)} "
                f"{'OK' if not attempt.reasons else f'FAIL {len(attempt.reasons)}'}"
            )
            append_usage(
                USAGE_LOG,
                {
                    "run_id": run_id,
                    "ts": datetime.now(UTC).isoformat(),
                    "job": job.key,
                    "source_type": job.source_type.value,
                    "gold_ids": job.gold_ids,
                    "model": deployment,
                    "attempt": attempt.attempt,
                    "dry_run": args.dry_run,
                    "ok": not attempt.reasons,
                    "evidence_ok": attempt.evidence_ok,
                    "reasons": attempt.reasons,
                    **usage.__dict__,
                    "cost_usd": cost,
                },
            )

        return log_call

    try:
        for job in jobs:
            ctx = JobContext.build(job, users, world, args.seed)
            number = next_number[job.source_type]
            result = render_job(
                client,
                deployment,
                ctx,
                world,
                users_by_id,
                number,
                log_call=make_logger(job),
            )
            if result.document is not None:
                totals.accepted += 1
                next_number[job.source_type] = number + 1
                print(f"  [OK] {result.document.doc_id} ← {job.key}")
                if args.dry_run:
                    print(result.document.model_dump_json(indent=2))
                else:
                    append_document(result.document, GENERATED_DIR)
            else:
                totals.rejected += 1
                print(f"  [REJECT] {job.key}")
                for r in result.attempts[-1].reasons:
                    print(f"      - {r[:300]}")
                if not args.dry_run:
                    save_rejected(result, REJECTED_DIR, run_id)
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
