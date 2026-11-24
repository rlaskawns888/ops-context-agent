"""mock 사용자 생성·검증 (LLM 호출 없음)."""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import generate_users
from common.schema.gold import GoldKnowledge, load_gold_knowledge, load_gold_yaml
from common.schema.users import MockUser, active_deny_rules, is_active, validate_users
from common.schema.world import load_world

WORLD = load_world(generate_users.WORLD_FILE)
GOLDS = load_gold_knowledge(generate_users.GOLD_DIR)
AS_OF = date(2026, 11, 24)


@pytest.fixture(scope="module")
def users() -> list[MockUser]:
    return generate_users.generate_users(WORLD, GOLDS, AS_OF)


def _user(**overrides: Any) -> MockUser:
    base = {
        "user_id": "U0001",
        "employee_no": "2020001",
        "name": "홍길동",
        "email": "gildong@opsctx.example",
        "phone": "010-1234-5678",
        "slack_user_id": "SABCD1234",
        "department": "개발1팀",
        "user_group": "정규직",
        "role": "requester",
        "it_team": None,
        "hire_date": date(2020, 3, 2),
        "account_status": "ACTIVE",
        "granted_permissions": [],
    }
    return MockUser.model_validate({**base, **overrides})


def _deny_gold(valid_from: date | None, valid_to: date | None) -> GoldKnowledge:
    """K003(외주 → 운영DB WRITE DENY)의 정책 기간만 바꾼 gold."""
    path = generate_users.GOLD_DIR / "K003.yaml"
    data = load_gold_yaml(path.read_text(encoding="utf-8"))
    for item in [*data["policies"], *(e for e in data["entities"] if e["type"] == "Policy")]:
        item["valid_from"], item["valid_to"] = valid_from, valid_to
    return GoldKnowledge.model_validate(data)


# -- 생성 ---------------------------------------------------------------------


def test_same_seed_same_result() -> None:
    first = generate_users.generate_users(WORLD, GOLDS, AS_OF, seed=42)
    second = generate_users.generate_users(WORLD, GOLDS, AS_OF, seed=42)
    assert [u.model_dump() for u in first] == [u.model_dump() for u in second]


def test_different_seed_different_result() -> None:
    first = generate_users.generate_users(WORLD, GOLDS, AS_OF, seed=42)
    other = generate_users.generate_users(WORLD, GOLDS, AS_OF, seed=7)
    assert [u.model_dump() for u in first] != [u.model_dump() for u in other]


def test_generated_users_pass_validation(users: list[MockUser]) -> None:
    assert len(users) == 200
    assert [u.user_id for u in users] == [f"U{i:04d}" for i in range(1, 201)]
    assert validate_users(users, WORLD, GOLDS, AS_OF, expected_count=200) == []


def test_it_staff_covers_every_team(users: list[MockUser]) -> None:
    staff = [u for u in users if u.role == "it_staff"]
    assert len(staff) == generate_users.IT_STAFF_COUNT
    for team in WORLD.teams:
        members = [u for u in staff if u.it_team == team.canonical_name]
        assert len(members) >= 2, team.canonical_name
        assert all(u.department == team.canonical_name for u in members)


def test_group_and_status_distribution(users: list[MockUser]) -> None:
    requesters = [u for u in users if u.role == "requester"]
    expected = generate_users.quota(generate_users.GROUP_SHARES, len(requesters))
    assert {g: sum(u.user_group == g for u in requesters) for g in expected} == expected
    assert sum(u.account_status == "LOCKED" for u in users) == 10
    assert sum(u.account_status == "DISABLED" for u in users) == 4


def test_new_hires_joined_after_september(users: list[MockUser]) -> None:
    for u in users:
        if u.user_group == "신규 입사자":
            assert u.hire_date >= date(2026, 9, 1)
        else:
            assert u.hire_date < date(2026, 9, 1)


def test_no_active_deny_permission_granted(users: list[MockUser]) -> None:
    denied = {(r.user_group, r.permission) for r in active_deny_rules(GOLDS, AS_OF)}
    assert denied  # gold에 유효한 DENY 정책이 있어야 의미가 있다
    for u in users:
        assert not {(u.user_group, p) for p in u.granted_permissions} & denied


def test_quota_sums_to_total() -> None:
    assert generate_users.quota(generate_users.GROUP_SHARES, 185) == {
        "정규직": 120,
        "외주": 37,
        "신규 입사자": 15,
        "시스템 관리자": 9,
        "임원": 4,
    }


def test_main_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "users.jsonl"
    assert generate_users.main(["--as-of", AS_OF.isoformat(), "--output", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    MockUser.model_validate_json(lines[0])


# -- 정책 검증 ------------------------------------------------------------------


def test_deny_policy_violation_fails() -> None:
    bad = _user(user_group="외주", granted_permissions=["운영DB WRITE"])
    errors = validate_users([bad], WORLD, [_deny_gold(None, None)], AS_OF)
    assert any("운영DB WRITE" in e and "K003" in e for e in errors)


def test_deny_policy_valid_to_is_inclusive() -> None:
    bad = _user(user_group="외주", granted_permissions=["운영DB WRITE"])
    gold = _deny_gold(date(2025, 1, 1), date(2026, 11, 30))
    assert validate_users([bad], WORLD, [gold], date(2026, 11, 30))
    assert validate_users([bad], WORLD, [gold], date(2026, 12, 1)) == []


def test_deny_policy_not_yet_valid_is_ignored() -> None:
    bad = _user(user_group="외주", granted_permissions=["운영DB WRITE"])
    gold = _deny_gold(date(2027, 1, 1), None)
    assert validate_users([bad], WORLD, [gold], AS_OF) == []


def test_deny_policy_other_group_not_affected() -> None:
    ok = _user(user_group="정규직", granted_permissions=["운영DB WRITE"])
    assert validate_users([ok], WORLD, [_deny_gold(None, None)], AS_OF) == []


def test_is_active_boundaries() -> None:
    d = date(2026, 1, 1)
    assert is_active(None, None, d)
    assert is_active(d, d, d)
    assert not is_active(date(2026, 1, 2), None, d)
    assert not is_active(None, date(2025, 12, 31), d)


# -- world 대조·고유값 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"user_group": "인턴"}, "user_group"),
        ({"granted_permissions": ["운영DB DROP"]}, "권한"),
        ({"role": "it_staff", "it_team": "없는팀"}, "it_team"),
    ],
)
def test_unknown_world_name_fails(overrides: dict[str, Any], needle: str) -> None:
    errors = validate_users([_user(**overrides)], WORLD, [], AS_OF)
    assert any(needle in e for e in errors)


@pytest.mark.parametrize("field", ["employee_no", "name", "email", "slack_user_id"])
def test_duplicate_unique_field_fails(field: str) -> None:
    a = _user()
    distinct = {
        "user_id": "U0002",
        "employee_no": "2020002",
        "name": "김철수",
        "email": "chulsoo@opsctx.example",
        "slack_user_id": "SZZZZ9999",
    }
    b = _user(**{**distinct, field: getattr(a, field)})
    errors = validate_users([a, b], WORLD, [], AS_OF)
    assert any(f"{field} 중복" in e for e in errors)


def test_wrong_count_fails() -> None:
    errors = validate_users([_user()], WORLD, [], AS_OF, expected_count=200)
    assert any("사용자 수" in e for e in errors)


# -- 모델 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "it_staff"},  # it_team 누락
        {"it_team": "DB운영팀"},  # requester인데 it_team
        {"employee_no": "2019001"},  # 입사연도 불일치
        {"granted_permissions": ["VPN 접속", "VPN 접속"]},
        {
            "granted_permissions": [
                "VPN 접속",
                "개발DB READ",
                "메일 외부발송",
                "DRM 해제",
                "AD 관리자",
            ]
        },
        {"email": "gildong@example.com"},
        {"slack_user_id": "U12345678"},
        {"account_status": "SUSPENDED"},
    ],
)
def test_invalid_user_model(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _user(**overrides)
