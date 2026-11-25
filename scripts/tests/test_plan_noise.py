"""노이즈 계획 (LLM 없음): 결정성·배분·gold 조합 회피·권한 신청 판정."""

from collections import Counter
from datetime import date

import pytest

import plan_noise as pn
from common.schema.gold import load_gold_knowledge
from common.schema.knowledge import PolicyEffect, SourceType
from common.schema.world import load_world

WORLD = load_world(pn.WORLD_FILE)
GOLDS = load_gold_knowledge(pn.GOLD_DIR)
USERS = pn.load_users(pn.USERS_FILE)
USERS_BY_ID = {u.user_id: u for u in USERS}


@pytest.fixture(scope="module")
def plan() -> list[pn.NoisePlanItem]:
    return pn.plan_noise(GOLDS, WORLD, USERS)


def test_same_seed_same_plan(plan):
    again = pn.plan_noise(GOLDS, WORLD, USERS, seed=pn.DEFAULT_SEED)
    assert pn.dump_plan(again) == pn.dump_plan(plan)
    assert pn.dump_plan(pn.plan_noise(GOLDS, WORLD, USERS, seed=7)) != pn.dump_plan(plan)


def test_plan_passes_validation(plan):
    assert pn.validate_plan(plan, GOLDS, WORLD, USERS) == []


def test_counts_and_ratios(plan):
    by_st = Counter(i.source_type for i in plan)
    assert {st: by_st.get(st, 0) for st in pn.NOISE_COUNTS} == pn.NOISE_COUNTS
    for st in (SourceType.SLACK, SourceType.TICKET):
        sub = [i for i in plan if i.source_type is st]
        cats = Counter(i.category for i in sub)
        assert dict(cats) == {
            k: v for k, v in pn.allocate(len(sub), pn.CATEGORY_WEIGHTS).items() if v
        }
    kinds = Counter((i.source_type, i.noise_kind) for i in plan)
    assert kinds[SourceType.TICKET, "unresolved"] == 0
    assert kinds[SourceType.SLACK, "unresolved"] == pytest.approx(27, abs=2)
    assert kinds[SourceType.SLACK, "repeat"] == pytest.approx(54, abs=2)
    assert all(i.noise_kind == "general" for i in plan if i.source_type is SourceType.GUIDE)
    # plan_id는 유형별 작성일 순 연번
    slack = [i for i in plan if i.source_type is SourceType.SLACK]
    assert [i.plan_id for i in slack[:2]] == ["NS-0001", "NS-0002"]
    assert [i.created_at for i in slack] == sorted(i.created_at for i in slack)


def test_dates_in_range_weekdays_and_recent_weighted(plan):
    days = [i.created_at.date() for i in plan]
    assert all(pn.DATE_MIN <= d <= pn.DATE_MAX for d in days)
    assert all(d.weekday() < 5 for d in days)
    general = [i.created_at.date() for i in plan if i.noise_kind != "repeat"]
    mid = pn.DATE_MIN + (pn.DATE_MAX - pn.DATE_MIN) / 2
    assert sum(d > mid for d in general) > sum(d <= mid for d in general)


def test_general_plans_avoid_gold_combinations(plan):
    gold_codes = pn.gold_error_codes(GOLDS)
    gold_pairs = pn.gold_policy_pairs(GOLDS)
    gold_errors = pn.gold_error_pairs(GOLDS)
    general = [i for i in plan if i.noise_kind != "repeat"]
    assert general
    for it in general:
        assert it.related_gold_ids == []
        assert it.error_code not in gold_codes, it
        assert (it.error_code, it.system) not in gold_errors, it
        if it.permission:
            group = USERS_BY_ID[it.requester_user_id].user_group
            assert (group, it.permission) not in gold_pairs, it
    assert {i.error_code for i in general} - {None} <= {
        "E-103",
        "E-201",
        "E-203",
        "E-301",
        "E-401",
        "E-402",
        "E-501",
        "E-552",
        "E-802",
    }


def test_repeat_plans_point_to_gold_of_same_category(plan):
    golds = {g.id: g for g in GOLDS}
    repeats = [i for i in plan if i.noise_kind == "repeat"]
    assert repeats
    for it in repeats:
        gold = golds[it.related_gold_ids[0]]
        assert pn.gold_category(gold) == it.category
        start, end = pn.date_window(gold, pn.gold_role_at(gold, it.created_at.date()))
        assert start <= it.created_at.date() <= end
        if not it.permission:
            assert it.error_code == pn.gold_code(gold)


@pytest.mark.parametrize(
    ("group", "permission", "day", "effect", "status"),
    [
        ("외주", "운영DB WRITE", date(2025, 6, 2), PolicyEffect.DENY, "REJECTED"),  # K003
        (
            "정규직",
            "EDR 예외 등록",
            date(2025, 12, 31),
            PolicyEffect.DENY,
            "REJECTED",
        ),  # K046 끝날 포함
        ("정규직", "EDR 예외 등록", date(2026, 1, 1), PolicyEffect.REQUIRE_APPROVAL, "RESOLVED"),
        ("정규직", "VPN 해외접속", date(2025, 12, 31), PolicyEffect.ALLOW, "RESOLVED"),  # K004
        ("신규 입사자", "메일 외부발송", date(2026, 11, 30), PolicyEffect.DENY, "REJECTED"),  # K055
        ("정규직", "망연계 반입", date(2025, 6, 2), None, "RESOLVED"),  # 정책 없음
    ],
)
def test_permission_judgement(group, permission, day, effect, status):
    got = pn.judge_permission(GOLDS, group, permission, day)
    assert got == effect
    assert pn.permission_status(got) == status


def test_permission_status_matches_gold_policy(plan):
    perm = [i for i in plan if i.permission and i.noise_kind != "unresolved"]
    assert any(i.status == "REJECTED" for i in perm)
    for it in perm:
        group = USERS_BY_ID[it.requester_user_id].user_group
        effect = pn.judge_permission(GOLDS, group, it.permission, it.created_at.date())
        assert it.status == pn.permission_status(effect), it
        if it.noise_kind == "repeat":  # 다시 묻는 gold 자신의 정책과도 판정이 같아야 한다
            gold = next(g for g in GOLDS if g.id == it.related_gold_ids[0])
            own = {
                p.effect
                for p in gold.policies
                if it.permission in pn.governed_permissions(gold, p.name)
                and pn.is_active(p.valid_from, p.valid_to, it.created_at.date())
            }
            assert own == {effect}, it


def test_validate_plan_catches_violations(plan):
    rejected = next(i for i in plan if i.status == "REJECTED")
    general = next(i for i in plan if i.noise_kind == "general" and i.system == "VPN")
    broken = [
        rejected.model_copy(update={"status": "RESOLVED"}),
        general.model_copy(update={"error_code": "E-102"}),  # gold(K001) 오류 코드
    ]
    errors = pn.validate_plan(broken, GOLDS, WORLD, USERS)
    assert any("권한 신청 status RESOLVED" in e for e in errors)
    assert any("gold 오류 코드 E-102" in e for e in errors)


def test_lock_category_prefers_locked_users(plan):
    lock = [i for i in plan if i.category == pn.LOCK_CATEGORY and i.source_type.value != "guide"]
    statuses = Counter(USERS_BY_ID[i.requester_user_id].account_status.value for i in lock)
    others = [i for i in plan if i.category != pn.LOCK_CATEGORY and i.requester_user_id]
    assert statuses["LOCKED"] + statuses["DISABLED"] > len(lock) / 2
    assert all(USERS_BY_ID[i.requester_user_id].account_status.value == "ACTIVE" for i in others)


def test_assignee_from_owning_team(plan):
    owner = {s.canonical_name: s.owner_team for s in WORLD.systems}
    for it in plan:
        if it.noise_kind != "repeat":
            assert USERS_BY_ID[it.assignee_user_id].it_team == owner[it.system], it


def test_allocate_and_sample_day():
    assert pn.allocate(270, pn.CATEGORY_WEIGHTS)["계정 잠금·비밀번호"] == 54
    assert sum(pn.allocate(477, pn.CATEGORY_WEIGHTS).values()) == 477
    assert pn.allocate(7, {"a": 1, "b": 1}) == {"a": 4, "b": 3}
    import random

    rng = random.Random(1)
    days = [pn.sample_day(rng, date(2025, 1, 1), date(2025, 1, 31)) for _ in range(200)]
    assert all(date(2025, 1, 1) <= d <= date(2025, 1, 31) and d.weekday() < 5 for d in days)
