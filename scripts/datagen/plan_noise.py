"""노이즈 문서 계획 (LLM 없음, 고정 seed).

실행:
    uv run python scripts/datagen/plan_noise.py            # data/generated/noise_plan.jsonl 생성
    uv run python scripts/datagen/plan_noise.py --force    # 기존 계획과 달라도 덮어쓰기

입력: data/gold/knowledge/*.yaml, data/world/world.yaml, data/generated/users.jsonl
출력: data/generated/noise_plan.jsonl (한 줄 = 노이즈 문서 1개의 시나리오)

gold 외 문서로 목표 규모(slack 300, ticket 500, guide 40, minutes 20)를 채우기 위한 계획이다.
무엇을·누가·언제·결과는 여기서 정하고, render_noise.py의 LLM은 문장만 쓴다.

- 생성 수: NOISE_COUNTS (slack 270, ticket 477, guide 8, minutes 0)
- category 비율(CATEGORY_WEIGHTS)을 slack·ticket 각각에 최대 잉여 방식으로 배분하고,
  category 안에서 noise_kind 비율(KIND_WEIGHTS)을 다시 배분한다.
  unresolved는 slack만 해당 → ticket은 general:repeat = 70:20으로 다시 나눈다.
- general/unresolved: gold가 쓰는 오류 코드·(Error, System) 조합·(UserGroup, Permission) 정책 조합
  사용 금지. 오류 코드는 world error_codes 중 gold가 쓰지 않은 것만 쓰고, 그런 코드가 없는
  시스템(DRM·EDR·백신·망연계 등)은 오류 코드 없는 일반 문의 주제(TOPICS)를 쓴다.
- repeat: 같은 category의 gold 1개를 다시 묻는다 (related_gold_ids). 날짜는 그 gold가 유효한
  범위 안. 권한 신청 repeat은 고른 gold 정책의 effect가 그 날짜의 전체 gold 판정과 같을 때만 쓴다
  (서로 다른 gold가 같은 조합에 다른 정책을 두는 기간을 피한다).
- 권한 신청 status: 요청자 user_group + permission을 작성일에 유효한 gold 정책으로 판정
  (valid_to는 마지막 날 포함, 여러 개면 DENY > REQUIRE_APPROVAL > ALLOW).
  DENY → REJECTED, REQUIRE_APPROVAL/ALLOW/정책 없음 → RESOLVED.
- 등장인물: 요청자는 role=requester. 계정 잠금·비밀번호는 LOCKED 사용자 우선
  (E-402 비활성화 문의는 DISABLED 우선), 그 외는 ACTIVE. 담당자는 해당 시스템 담당 팀
  (repeat은 gold OWNED_BY, 없으면 world owner_team)의 it_staff.
- 날짜: DATE_MIN ~ DATE_MAX 평일 업무 시간, 밀도가 최근으로 갈수록 선형 증가(RECENCY).
- plan_id는 source_type별 작성일 순으로 NS-0001(slack), NT-0001(ticket), NG-001(guide).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from common.schema.documents import NoiseKind
from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.knowledge import (
    EntityType,
    ExtractedPolicy,
    PolicyEffect,
    RelationType,
    SourceType,
)
from common.schema.users import AccountStatus, MockUser, is_active
from common.schema.world import World, load_world
from gold_rules import DATE_MAX, DATE_MIN
from render_docs import (
    GENERATED_DIR,
    GOLD_DIR,
    USERS_FILE,
    WORLD_FILE,
    Role,
    date_window,
    load_users,
    primary_systems,
    responsible_teams,
    revision_policy_names,
)

PLAN_FILE = GENERATED_DIR / "noise_plan.jsonl"
DEFAULT_SEED = 42

NOISE_COUNTS: dict[SourceType, int] = {
    SourceType.SLACK: 270,
    SourceType.TICKET: 477,
    SourceType.GUIDE: 8,
    SourceType.MINUTES: 0,
}

Category = Literal[
    "계정 잠금·비밀번호",
    "VPN",
    "MFA·SSO 인증",
    "보안 프로그램",
    "권한 신청",
    "메일·그룹웨어",
    "망연계",
    "기타",
]
PERMISSION_CATEGORY: Category = "권한 신청"
LOCK_CATEGORY: Category = "계정 잠금·비밀번호"
CATEGORY_WEIGHTS: dict[Category, int] = {
    "계정 잠금·비밀번호": 20,
    "VPN": 15,
    "MFA·SSO 인증": 15,
    "보안 프로그램": 15,
    "권한 신청": 15,
    "메일·그룹웨어": 8,
    "망연계": 7,
    "기타": 5,
}
KIND_WEIGHTS: dict[NoiseKind, int] = {"general": 70, "repeat": 20, "unresolved": 10}

# 시스템 → category (권한 신청은 시스템이 아니라 permission 유무로 정한다)
SYSTEM_CATEGORY: dict[str, Category] = {
    "AD 계정": "계정 잠금·비밀번호",
    "VPN": "VPN",
    "MFA": "MFA·SSO 인증",
    "SSO": "MFA·SSO 인증",
    "DRM": "보안 프로그램",
    "EDR": "보안 프로그램",
    "백신": "보안 프로그램",
    "메일": "메일·그룹웨어",
    "그룹웨어": "메일·그룹웨어",
    "망연계": "망연계",
    "사내 포털": "기타",
    "티켓 시스템": "기타",
    "운영DB": "기타",
    "개발DB": "기타",
    "형상관리": "기타",
    "배포시스템": "기타",
}

# gold가 쓰지 않았지만 일반 문의로 쓰면 gold 정책과 부딪히는 코드.
# E-104(해외 접속 차단)는 해외 VPN 접속 정책(gold) 판단이 따라붙어 general에서 뺀다.
EXCLUDED_GENERAL_CODES = frozenset({"E-104"})

# 오류 코드 없는 일반 문의 주제 (category → (system, topic)). gold 주제와 겹치지 않게 골랐다.
TOPICS: dict[Category, list[tuple[str, str]]] = {
    "계정 잠금·비밀번호": [
        ("AD 계정", "비밀번호 변경 규칙(길이·조합·재사용 제한) 문의"),
        ("AD 계정", "비밀번호를 바꾼 뒤 휴대폰 메일 앱에 새 비밀번호를 다시 넣는 방법"),
        ("AD 계정", "비밀번호 만료 예정 안내 메일이 진짜 사내 메일인지 확인 요청"),
    ],
    "VPN": [
        ("VPN", "새 노트북에 VPN 클라이언트를 설치하는 방법"),
        ("VPN", "VPN 연결 후 사내 시스템 화면이 느리게 뜸"),
        ("VPN", "VPN 연결 중에는 집 프린터가 잡히지 않음"),
    ],
    "MFA·SSO 인증": [
        ("SSO", "SSO 자동 로그아웃까지 걸리는 시간 문의"),
        ("MFA", "MFA 앱 승인 알림의 소리·진동 설정 방법"),
        ("SSO", "SSO로 들어가는 사내 시스템 목록 문의"),
    ],
    "보안 프로그램": [
        ("백신", "백신 정기 정밀 검사 시간을 바꾸는 방법"),
        ("백신", "백신 검사 중 PC가 느려짐"),
        ("EDR", "EDR 에이전트가 PC에 설치되어 있는지 확인하는 방법"),
        ("DRM", "DRM 적용 문서를 인쇄하면 워터마크가 찍히는 이유"),
        ("DRM", "새로 만든 엑셀 파일에도 DRM이 자동 적용되는지 문의"),
    ],
    "메일·그룹웨어": [
        ("메일", "메일 서명 설정 방법"),
        ("메일", "휴가 중 부재중 자동응답 설정"),
        ("그룹웨어", "그룹웨어 일정을 팀원에게 공유하는 방법"),
        ("그룹웨어", "상신한 결재 문서를 회수하는 방법"),
        ("메일", "거래처 메일이 스팸함으로 들어감"),
    ],
    "망연계": [
        ("망연계", "망연계 전송 이력 조회 방법"),
        ("망연계", "망연계로 보낸 파일의 보관 기간 문의"),
        ("망연계", "인터넷망 PC에서 망연계 접속 주소 문의"),
        ("망연계", "망연계 전송 대기 시간이 평소보다 길어짐"),
    ],
    "기타": [
        ("사내 포털", "사내 포털 공지 게시 요청 방법"),
        ("티켓 시스템", "내가 올린 요청의 처리 현황을 조회하는 방법"),
        ("개발DB", "개발DB 접속 툴 설치 문의"),
        ("형상관리", "형상관리에 새 저장소 생성 요청"),
        ("배포시스템", "개발계 배포 일정 확인 방법"),
        ("사내 포털", "포털 첫 화면 바로가기 편집 방법"),
    ],
}


@dataclass(frozen=True)
class GuideTopic:
    category: Category
    system: str
    error_code: str | None
    topic: str


GUIDE_TOPICS: list[GuideTopic] = [
    GuideTopic(
        "기타", "사내 포털", None, "신규 입사자 IT 체크리스트 (첫 출근 주 계정·PC·포털 준비)"
    ),
    GuideTopic("메일·그룹웨어", "메일", "E-501", "메일함 용량 정리 방법"),
    GuideTopic("계정 잠금·비밀번호", "AD 계정", "E-401", "비밀번호 설정 규칙과 정기 변경 방법"),
    GuideTopic("메일·그룹웨어", "그룹웨어", None, "그룹웨어 일정·회의실 예약 이용 안내"),
    GuideTopic("VPN", "VPN", "E-103", "재택근무 시 VPN 사용 수칙"),
    GuideTopic("보안 프로그램", "백신", None, "PC 백신 정기 검사 설정 방법"),
    GuideTopic("망연계", "망연계", None, "망연계 전송 이력 조회 방법"),
    GuideTopic("기타", "티켓 시스템", None, "티켓 시스템 요청 처리 현황 조회 방법"),
]

PLAN_ID_PREFIX: dict[SourceType, tuple[str, int]] = {
    SourceType.SLACK: ("NS", 4),
    SourceType.TICKET: ("NT", 4),
    SourceType.GUIDE: ("NG", 3),
    SourceType.MINUTES: ("NM", 3),
}
RECENCY = 0.5  # 날짜 밀도: 시작 1.0 → 끝 1.5
RESOLVE_MARGIN_DAYS = 3  # 댓글·해결 시각이 범위를 넘지 않게 작성일 뒤에 남기는 여유
LOCKED_PREFERENCE = 0.7  # 계정 잠금·비밀번호 요청자를 LOCKED(E-402는 DISABLED)에서 뽑을 확률
RISK_WEIGHT = {"LOW": 3, "MEDIUM": 2, "HIGH": 1}  # general 권한 신청에서 권한 선택 가중치
MAX_DATE_TRIES = 200

PlanStatus = Literal["RESOLVED", "REJECTED", "UNRESOLVED"]
UnresolvedMode = Literal["no_answer", "will_check"]


class NoisePlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    source_type: SourceType
    category: Category
    noise_kind: NoiseKind
    system: str
    error_code: str | None
    permission: str | None
    topic: str
    requester_user_id: str | None  # guide는 None
    assignee_user_id: str  # guide는 작성 부서 담당자
    created_at: datetime
    status: PlanStatus | None  # guide는 None
    unresolved_mode: UnresolvedMode | None
    related_gold_ids: list[str]


# ---------------------------------------------------------------------------
# gold 판정
# ---------------------------------------------------------------------------


def governed_permissions(gold: GoldKnowledge, policy_name: str) -> list[str]:
    return [
        r.target_name
        for r in gold.relations
        if r.relation_type is RelationType.GOVERNS
        and r.source_type is EntityType.POLICY
        and r.source_name == policy_name
        and r.target_type is EntityType.PERMISSION
    ]


_EFFECT_ORDER = [PolicyEffect.DENY, PolicyEffect.REQUIRE_APPROVAL, PolicyEffect.ALLOW]


def judge_permission(
    golds: Iterable[GoldKnowledge], user_group: str, permission: str, as_of: date
) -> PolicyEffect | None:
    """as_of에 유효한 gold 정책으로 (user_group, permission) 판정. 정책이 없으면 None."""
    effects = {
        p.effect
        for g in golds
        for p in g.policies
        if p.subject_group == user_group
        and permission in governed_permissions(g, p.name)
        and is_active(p.valid_from, p.valid_to, as_of)
    }
    return next((e for e in _EFFECT_ORDER if e in effects), None)


def permission_status(effect: PolicyEffect | None) -> PlanStatus:
    return "REJECTED" if effect is PolicyEffect.DENY else "RESOLVED"


def gold_policy_pairs(golds: Iterable[GoldKnowledge]) -> set[tuple[str, str]]:
    """gold 정책이 다루는 (UserGroup, Permission) 조합 (기간 무관)."""
    return {
        (p.subject_group, perm)
        for g in golds
        for p in g.policies
        for perm in governed_permissions(g, p.name)
    }


def gold_error_codes(golds: Iterable[GoldKnowledge]) -> set[str]:
    return {e.code for g in golds for e in g.entities if e.type is EntityType.ERROR and e.code}


def gold_error_pairs(golds: Iterable[GoldKnowledge]) -> set[tuple[str, str]]:
    """gold의 (Error canonical_name, System) 조합 (OCCURS_IN)."""
    return {
        (r.source_name, r.target_name)
        for g in golds
        for r in g.relations
        if r.relation_type is RelationType.OCCURS_IN
    }


def gold_error_system(gold: GoldKnowledge) -> str | None:
    return next(
        (r.target_name for r in gold.relations if r.relation_type is RelationType.OCCURS_IN),
        None,
    )


def gold_category(gold: GoldKnowledge) -> Category:
    if gold.policies:
        return PERMISSION_CATEGORY
    system = gold_error_system(gold) or primary_systems(gold)[0]
    return SYSTEM_CATEGORY[system]


def gold_code(gold: GoldKnowledge) -> str | None:
    return next((e.code for e in gold.entities if e.type is EntityType.ERROR and e.code), None)


def gold_role_at(gold: GoldKnowledge, day: date) -> Role:
    """day 시점에 이 gold의 어느 몫이 유효한가 (policy_revision이 아니면 full)."""
    if gold.scenario_type != "policy_revision":
        return "full"
    _, new_name = revision_policy_names(gold)
    new = next(p for p in gold.policies if p.name == new_name)
    assert new.valid_from is not None
    return "old" if day < new.valid_from else "new"


def gold_window_at(gold: GoldKnowledge, day: date) -> tuple[date, date]:
    return date_window(gold, gold_role_at(gold, day))


# ---------------------------------------------------------------------------
# 배분·추출 도구
# ---------------------------------------------------------------------------


def allocate[K](total: int, weights: Mapping[K, int]) -> dict[K, int]:
    """total을 weights 비율로 정수 배분 (최대 잉여 방식, 동률은 앞 키 우선)."""
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    rest = total - sum(out.values())
    order = sorted(weights, key=lambda k: -(raw[k] - out[k]))  # sorted는 안정 정렬
    for k in order[:rest]:
        out[k] += 1
    return out


class Cycler[T]:
    """목록을 섞어 차례로 내주고, 다 쓰면 다시 섞는다 (고르게 돌려 쓰기)."""

    def __init__(self, items: Sequence[T], rng: random.Random) -> None:
        if not items:
            raise ValueError("빈 목록은 돌릴 수 없습니다")
        self.items, self.rng, self.queue = list(items), rng, []

    def next(self) -> T:
        if not self.queue:
            self.queue = list(self.items)
            self.rng.shuffle(self.queue)
        return self.queue.pop()


def sample_day(rng: random.Random, start: date, end: date) -> date:
    """[start, end]의 평일 하나. 밀도는 start → end로 1 → 1+RECENCY 선형 증가."""
    if start > end:
        raise ValueError(f"날짜 범위가 비어 있습니다: {start} ~ {end}")
    span = (end - start).days
    u = rng.random()
    a = RECENCY / 2  # CDF(t) ∝ t + a·t²
    t = (-1 + math.sqrt(1 + 4 * a * u * (1 + a))) / (2 * a)
    day = start + timedelta(days=min(span, round(t * span)))
    if day.weekday() >= 5:
        back = day - timedelta(days=day.weekday() - 4)
        fwd = day + timedelta(days=7 - day.weekday())
        if back >= start:
            day = back
        elif fwd <= end:
            day = fwd
    return day


def business_time(rng: random.Random, day: date) -> datetime:
    return datetime.combine(day, dtime(rng.randint(9, 17), rng.randrange(0, 60, 5)))


def resolve_end(end: date) -> date:
    return end - timedelta(days=RESOLVE_MARGIN_DAYS)


# ---------------------------------------------------------------------------
# 계획
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralOption:
    system: str
    error_code: str | None
    topic: str


@dataclass(frozen=True)
class PermissionRepeat:
    gold: GoldKnowledge
    policy: ExtractedPolicy
    permission: str


class Planner:
    def __init__(
        self,
        golds: Sequence[GoldKnowledge],
        world: World,
        users: Sequence[MockUser],
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.golds = sorted(golds, key=lambda g: g.id)
        self.world = world
        self.users = list(users)
        self.rng = random.Random(seed)
        self.gold_codes = gold_error_codes(self.golds)
        self.gold_pairs = gold_policy_pairs(self.golds)
        self.owner = {s.canonical_name: s.owner_team for s in world.systems}
        self.permissions = {p.canonical_name: p for p in world.permissions}
        self.requesters = [u for u in self.users if u.role == "requester"]
        self.staff = [u for u in self.users if u.role == "it_staff"]
        self._general: dict[Category, Cycler[GeneralOption]] = {}
        self._repeat: dict[Category, Cycler[Any]] = {}

    # -- 선택지 --------------------------------------------------------------

    def general_options(self, category: Category) -> list[GeneralOption]:
        codes = [
            GeneralOption(e.system, e.code, e.message)
            for e in self.world.error_codes
            if SYSTEM_CATEGORY[e.system] == category
            and e.code not in self.gold_codes
            and e.code not in EXCLUDED_GENERAL_CODES
        ]
        topics = [GeneralOption(s, None, t) for s, t in TOPICS[category]]
        return codes + topics

    def general_cycler(self, category: Category) -> Cycler[GeneralOption]:
        if category not in self._general:
            self._general[category] = Cycler(self.general_options(category), self.rng)
        return self._general[category]

    def repeat_cycler(self, category: Category) -> Cycler[Any]:
        if category not in self._repeat:
            golds = [g for g in self.golds if gold_category(g) == category]
            if category == PERMISSION_CATEGORY:
                items: list[Any] = [
                    PermissionRepeat(g, p, perm)
                    for g in golds
                    for p in g.policies
                    for perm in governed_permissions(g, p.name)
                ]
            else:
                items = golds
            self._repeat[category] = Cycler(items, self.rng)
        return self._repeat[category]

    # -- 사람 ----------------------------------------------------------------

    def pick_staff_for_teams(self, teams: Sequence[str]) -> MockUser:
        pool = [u for u in self.staff if u.it_team in teams] or self.staff
        return self.rng.choice(pool)

    def pick_requester(self, category: Category, error_code: str | None) -> MockUser:
        active = [u for u in self.requesters if u.account_status is AccountStatus.ACTIVE]
        if category == LOCK_CATEGORY:
            preferred = AccountStatus.DISABLED if error_code == "E-402" else AccountStatus.LOCKED
            pool = [u for u in self.requesters if u.account_status is preferred]
            if pool and self.rng.random() < LOCKED_PREFERENCE:
                return self.rng.choice(pool)
        return self.rng.choice(active)

    # -- 항목 ----------------------------------------------------------------

    def _item(self, st: SourceType, category: Category, kind: NoiseKind, **kw: Any) -> dict:
        unresolved_mode = None
        status = kw.pop("status", "RESOLVED")
        if kind == "unresolved":
            status = "UNRESOLVED"
            unresolved_mode = self.rng.choice(["no_answer", "will_check"])
        return {
            "plan_id": "",
            "source_type": st,
            "category": category,
            "noise_kind": kind,
            "status": status,
            "unresolved_mode": unresolved_mode,
            "related_gold_ids": [],
            **kw,
        }

    def general_item(self, st: SourceType, category: Category, kind: NoiseKind) -> dict:
        if category == PERMISSION_CATEGORY:
            return self.general_permission_item(st, kind)
        opt = self.general_cycler(category).next()
        requester = self.pick_requester(category, opt.error_code)
        day = sample_day(self.rng, DATE_MIN, resolve_end(DATE_MAX))
        return self._item(
            st,
            category,
            kind,
            system=opt.system,
            error_code=opt.error_code,
            permission=None,
            topic=opt.topic,
            requester_user_id=requester.user_id,
            assignee_user_id=self.pick_staff_for_teams([self.owner[opt.system]]).user_id,
            created_at=business_time(self.rng, day),
        )

    def general_permission_item(self, st: SourceType, kind: NoiseKind) -> dict:
        names = list(self.permissions)
        weights = [RISK_WEIGHT[self.permissions[n].risk_level.value] for n in names]
        active = [u for u in self.requesters if u.account_status is AccountStatus.ACTIVE]
        for _ in range(100):
            perm = self.rng.choices(names, weights)[0]
            pool = [
                u
                for u in active
                if (u.user_group, perm) not in self.gold_pairs and perm not in u.granted_permissions
            ]
            if pool:
                break
        else:
            raise ValueError("gold 정책과 겹치지 않는 권한 신청 조합을 찾지 못했습니다")
        requester = self.rng.choice(pool)
        day = sample_day(self.rng, DATE_MIN, resolve_end(DATE_MAX))
        system = self.permissions[perm].on_system
        status = permission_status(judge_permission(self.golds, requester.user_group, perm, day))
        return self._item(
            st,
            PERMISSION_CATEGORY,
            kind,
            system=system,
            error_code=None,
            permission=perm,
            topic=f"{perm} 권한 신청",
            requester_user_id=requester.user_id,
            assignee_user_id=self.pick_staff_for_teams([self.owner[system]]).user_id,
            created_at=business_time(self.rng, day),
            status=status,
        )

    def repeat_item(self, st: SourceType, category: Category) -> dict:
        if category == PERMISSION_CATEGORY:
            return self.repeat_permission_item(st)
        gold: GoldKnowledge = self.repeat_cycler(category).next()
        start, end = date_window(gold, "full")
        day = sample_day(self.rng, start, max(start, resolve_end(end)))
        system = gold_error_system(gold) or primary_systems(gold)[0]
        code = gold_code(gold)
        requester = self.pick_requester(category, code)
        return self._item(
            st,
            category,
            "repeat",
            system=system,
            error_code=code,
            permission=None,
            topic=gold.title.split(" (")[0],
            requester_user_id=requester.user_id,
            assignee_user_id=self.pick_staff_for_teams(responsible_teams(gold, self.world)).user_id,
            created_at=business_time(self.rng, day),
            related_gold_ids=[gold.id],
        )

    def repeat_permission_item(self, st: SourceType) -> dict:
        cycler = self.repeat_cycler(PERMISSION_CATEGORY)
        for _ in range(MAX_DATE_TRIES):
            cand: PermissionRepeat = cycler.next()
            p = cand.policy
            start = max(DATE_MIN, p.valid_from or DATE_MIN)
            end = resolve_end(min(DATE_MAX, p.valid_to or DATE_MAX))
            if start > end:
                continue
            day = sample_day(self.rng, start, end)
            effect = judge_permission(self.golds, p.subject_group, cand.permission, day)
            if effect != p.effect:
                continue  # 다른 gold 정책이 같은 날 다른 판정을 내림 → 이 날짜는 쓰지 않는다
            group = [
                u
                for u in self.requesters
                if u.user_group == p.subject_group and u.account_status is AccountStatus.ACTIVE
            ]
            pool = [u for u in group if cand.permission not in u.granted_permissions] or group
            if not pool:
                continue
            requester = self.rng.choice(pool)
            return self._item(
                st,
                PERMISSION_CATEGORY,
                "repeat",
                system=p.system_name,
                error_code=None,
                permission=cand.permission,
                topic=f"{cand.permission} 권한 신청",
                requester_user_id=requester.user_id,
                assignee_user_id=self.pick_staff_for_teams(
                    responsible_teams(cand.gold, self.world)
                ).user_id,
                created_at=business_time(self.rng, day),
                status=permission_status(effect),
                related_gold_ids=[cand.gold.id],
            )
        raise ValueError("권한 신청 repeat 날짜를 정하지 못했습니다")

    def guide_item(self, topic: GuideTopic) -> dict:
        day = sample_day(self.rng, DATE_MIN, DATE_MAX)
        return {
            "plan_id": "",
            "source_type": SourceType.GUIDE,
            "category": topic.category,
            "noise_kind": "general",
            "system": topic.system,
            "error_code": topic.error_code,
            "permission": None,
            "topic": topic.topic,
            "requester_user_id": None,
            "assignee_user_id": self.pick_staff_for_teams([self.owner[topic.system]]).user_id,
            "created_at": datetime.combine(day, dtime(9, 0)),
            "status": None,
            "unresolved_mode": None,
            "related_gold_ids": [],
        }

    # -- 전체 ----------------------------------------------------------------

    def plan_source(self, st: SourceType, count: int) -> list[dict]:
        if st is SourceType.GUIDE:
            if count > len(GUIDE_TOPICS):
                raise ValueError(f"guide 주제가 {len(GUIDE_TOPICS)}개뿐입니다")
            return [self.guide_item(t) for t in GUIDE_TOPICS[:count]]
        if st is SourceType.MINUTES:
            if count:
                raise ValueError("minutes 노이즈는 계획하지 않습니다")
            return []
        kinds = dict(KIND_WEIGHTS)
        if st is not SourceType.SLACK:
            kinds.pop("unresolved")
        out: list[dict] = []
        for category, n in allocate(count, CATEGORY_WEIGHTS).items():
            for kind, m in allocate(n, kinds).items():
                for _ in range(m):
                    if kind == "repeat":
                        out.append(self.repeat_item(st, category))
                    else:
                        out.append(self.general_item(st, category, kind))
        return out

    def plan(self, counts: Mapping[SourceType, int] = NOISE_COUNTS) -> list[NoisePlanItem]:
        items: list[NoisePlanItem] = []
        for st, count in counts.items():
            raw = self.plan_source(st, count)
            raw.sort(key=lambda d: d["created_at"])  # 안정 정렬: 같은 시각은 생성 순
            prefix, width = PLAN_ID_PREFIX[st]
            for i, d in enumerate(raw, 1):
                d["plan_id"] = f"{prefix}-{i:0{width}d}"
                items.append(NoisePlanItem.model_validate(d))
        return items


def plan_noise(
    golds: Sequence[GoldKnowledge],
    world: World,
    users: Sequence[MockUser],
    seed: int = DEFAULT_SEED,
    counts: Mapping[SourceType, int] = NOISE_COUNTS,
) -> list[NoisePlanItem]:
    return Planner(golds, world, users, seed).plan(counts)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def validate_plan(
    items: Sequence[NoisePlanItem],
    golds: Sequence[GoldKnowledge],
    world: World,
    users: Sequence[MockUser],
) -> list[str]:
    """계획 규칙 위반 사유 목록 (비어 있으면 통과)."""
    errors: list[str] = []
    by_user = {u.user_id: u for u in users}
    by_gold = {g.id: g for g in golds}
    systems = {s.canonical_name: s for s in world.systems}
    codes = {e.code: e for e in world.error_codes}
    perms = {p.canonical_name: p for p in world.permissions}
    gcodes, gpairs = gold_error_codes(golds), gold_policy_pairs(golds)

    ids = Counter(i.plan_id for i in items)
    errors += [f"plan_id 중복: {k}" for k, n in ids.items() if n > 1]

    for it in items:
        e = errors.append
        tag = it.plan_id
        if it.system not in systems:
            e(f"{tag}: 시스템 {it.system!r}가 world에 없음")
        if it.error_code is not None:
            if it.error_code not in codes:
                e(f"{tag}: 오류 코드 {it.error_code}가 world에 없음")
            elif codes[it.error_code].system != it.system:
                e(f"{tag}: 오류 코드 {it.error_code}의 시스템이 {it.system}이 아님")
        if it.permission is not None and it.permission not in perms:
            e(f"{tag}: 권한 {it.permission!r}가 world에 없음")
        if not DATE_MIN <= it.created_at.date() <= DATE_MAX:
            e(f"{tag}: 작성일 {it.created_at.date()}가 범위 밖")
        if it.noise_kind == "unresolved" and it.source_type is not SourceType.SLACK:
            e(f"{tag}: unresolved는 slack만")
        if (it.status == "UNRESOLVED") != (it.noise_kind == "unresolved"):
            e(f"{tag}: UNRESOLVED status와 noise_kind 불일치")

        assignee = by_user.get(it.assignee_user_id)
        if assignee is None or assignee.role != "it_staff":
            e(f"{tag}: 담당자 {it.assignee_user_id}가 it_staff가 아님")
        requester = by_user.get(it.requester_user_id or "")
        if it.source_type is SourceType.GUIDE:
            if it.requester_user_id is not None or it.status is not None:
                e(f"{tag}: guide는 요청자·status가 없어야 함")
        elif requester is None or requester.role != "requester":
            e(f"{tag}: 요청자 {it.requester_user_id}가 requester가 아님")

        if it.noise_kind == "repeat":
            if len(it.related_gold_ids) != 1 or it.related_gold_ids[0] not in by_gold:
                e(f"{tag}: repeat은 gold 1개를 가리켜야 함")
        else:
            if it.related_gold_ids:
                e(f"{tag}: {it.noise_kind}에 related_gold_ids가 있음")
            if it.error_code in gcodes:
                e(f"{tag}: gold 오류 코드 {it.error_code} 사용")
            if requester and it.permission and (requester.user_group, it.permission) in gpairs:
                e(f"{tag}: gold 정책 조합 ({requester.user_group}, {it.permission}) 사용")

        if it.permission is not None and requester is not None and it.noise_kind != "unresolved":
            day = it.created_at.date()
            want = permission_status(
                judge_permission(golds, requester.user_group, it.permission, day)
            )
            if it.status != want:
                e(f"{tag}: 권한 신청 status {it.status} (정책 판정 {want})")
    return errors


# ---------------------------------------------------------------------------
# 입출력·CLI
# ---------------------------------------------------------------------------


def dump_plan(items: Sequence[NoisePlanItem]) -> str:
    return "".join(i.model_dump_json() + "\n" for i in items)


def load_plan(path: Path = PLAN_FILE) -> list[NoisePlanItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [NoisePlanItem.model_validate_json(line) for line in lines if line.strip()]


def summarize(items: Sequence[NoisePlanItem]) -> str:
    lines = []
    for st in NOISE_COUNTS:
        sub = [i for i in items if i.source_type is st]
        if not sub:
            continue
        lines.append(f"[{st.value}] {len(sub)}건")
        for label, key in (
            ("category", lambda i: i.category),
            ("noise_kind", lambda i: i.noise_kind),
            ("status", lambda i: i.status),
            ("연도", lambda i: i.created_at.year),
        ):
            c = Counter(key(i) for i in sub)
            lines.append(f"  {label}: " + ", ".join(f"{k} {n}" for k, n in c.items()))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="노이즈 문서 계획 생성 (LLM 없음)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=PLAN_FILE)
    parser.add_argument("--force", action="store_true", help="기존 계획과 달라도 덮어쓰기")
    args = parser.parse_args(argv)

    world = load_world(WORLD_FILE)
    golds = load_gold_knowledge(GOLD_DIR)
    users = load_users(USERS_FILE)
    items = plan_noise(golds, world, users, args.seed)
    errors = validate_plan(items, golds, world, users)
    if errors:
        print("\n".join(f"  - {e}" for e in errors[:50]))
        raise SystemExit(f"계획 검증 실패 {len(errors)}건")

    text = dump_plan(items)
    if args.out.exists() and args.out.read_text(encoding="utf-8") != text and not args.force:
        raise SystemExit(
            f"{args.out}가 이미 있고 새 계획과 다릅니다. 렌더링이 진행 중이면 plan_id가 어긋나니 "
            "확인 후 --force로 덮어쓰세요."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(summarize(items))
    print(f"\n저장: {args.out} ({len(items)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
