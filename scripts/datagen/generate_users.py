"""mock 사용자 200명 생성기 (LLM 호출 없음).

실행:
    uv run python scripts/datagen/generate_users.py                    # 기준일 = 오늘
    uv run python scripts/datagen/generate_users.py --as-of 2026-11-24

흐름
1. world.yaml·gold를 읽는다.
2. Faker("ko_KR")·random.Random을 같은 seed(기본 42)로 초기화해 사용자를 만든다.
   - it_staff 15명: world teams마다 최소 2명, 팀마다 첫 1명은 시스템 관리자 그룹
   - requester 185명: user_group은 GROUP_SHARES 비율(최대 잔여 배분)로 정확히 나눈다
   - account_status: 전체 대비 LOCKED 5%·DISABLED 2%를 requester 중에서 고른다
   - granted_permissions: 부서·그룹·IT팀 후보 중 0~4개. 기준일에 유효한 gold DENY 정책이
     금지하는 권한은 후보에서 뺀다.
3. common.schema.users.validate_users로 검증하고, 통과해야만 data/generated/users.jsonl에 저장.
4. 분포 요약을 출력한다.

같은 seed·같은 기준일·같은 world/gold면 몇 번 실행해도 결과가 같다.
기준일이 바뀌어 유효한 DENY 정책 집합이 달라지면 권한 후보가 달라질 수 있다.
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.users import (
    AccountStatus,
    MockUser,
    UserRole,
    active_deny_rules,
    validate_users,
)
from common.schema.world import World, load_world

ROOT = Path(__file__).resolve().parents[2]
WORLD_FILE = ROOT / "data" / "world" / "world.yaml"
GOLD_DIR = ROOT / "data" / "gold" / "knowledge"
OUTPUT_FILE = ROOT / "data" / "generated" / "users.jsonl"

SEED = 42
TOTAL_USERS = 200
IT_STAFF_COUNT = 15
MIN_STAFF_PER_TEAM = 2
EMAIL_DOMAIN = "opsctx.example"

# requester user_group 비율 (world canonical_name 그대로)
GROUP_SHARES: dict[str, float] = {
    "정규직": 0.65,
    "외주": 0.20,
    "신규 입사자": 0.08,
    "시스템 관리자": 0.05,
    "임원": 0.02,
}
# 전체 인원 대비 비율. 나머지는 ACTIVE
STATUS_SHARES: dict[AccountStatus, float] = {
    AccountStatus.LOCKED: 0.05,
    AccountStatus.DISABLED: 0.02,
}

BUSINESS_DEPARTMENTS = ["영업", "재무", "인사", "마케팅", "개발1팀", "개발2팀", "고객지원", "기획"]
# 그룹별로 배치될 수 있는 업무 부서 (없으면 전체)
GROUP_DEPARTMENTS: dict[str, list[str]] = {
    "외주": ["개발1팀", "개발2팀", "고객지원", "기획"],
    "시스템 관리자": ["개발1팀", "개발2팀"],
}

# 입사일 범위 (시작, 끝 포함)
NEW_HIRE_FROM = date(2026, 9, 1)
NEW_HIRE_TO = date(2026, 11, 20)
HIRE_RANGES: dict[str, tuple[date, date]] = {
    "신규 입사자": (NEW_HIRE_FROM, NEW_HIRE_TO),
    "임원": (date(2005, 1, 1), date(2016, 12, 31)),
    "외주": (date(2021, 1, 1), date(2026, 8, 31)),
}
DEFAULT_HIRE_RANGE = (date(2008, 1, 1), date(2026, 8, 31))
IT_STAFF_HIRE_RANGE = (date(2010, 1, 1), date(2025, 12, 31))

# 권한 후보 (world permissions canonical_name). 순서가 결과에 영향을 주므로 list로 고정.
DEPARTMENT_PERMISSIONS: dict[str, list[str]] = {
    "영업": ["메일 외부발송", "메일 대용량첨부", "VPN 접속"],
    "재무": ["메일 외부발송", "망연계 반출", "VPN 접속", "운영DB READ"],
    "인사": ["메일 외부발송", "메일 대용량첨부", "DRM 해제", "전자결재 관리자"],
    "마케팅": ["메일 외부발송", "메일 대용량첨부", "망연계 반입", "VPN 접속"],
    "개발1팀": ["형상관리 READ", "형상관리 WRITE", "개발DB READ", "개발DB WRITE", "개발계 배포"],
    "개발2팀": ["형상관리 READ", "형상관리 WRITE", "개발DB READ", "개발DB WRITE", "VPN 접속"],
    "고객지원": ["운영DB READ", "메일 외부발송", "VPN 접속"],
    "기획": ["메일 외부발송", "메일 대용량첨부", "망연계 반입", "전자결재 관리자"],
}
GROUP_PERMISSIONS: dict[str, list[str]] = {
    "시스템 관리자": ["운영DB READ", "운영DB WRITE", "운영계 배포"],
    "임원": ["VPN 해외접속", "메일 외부발송", "백신 검사 예외"],
}
TEAM_PERMISSIONS: dict[str, list[str]] = {
    "보안운영팀": ["EDR 예외 등록", "백신 검사 예외", "DRM 해제", "VPN 접속"],
    "네트워크팀": ["VPN 접속", "VPN 해외접속", "망연계 반입", "망연계 반출"],
    "인프라운영팀": ["AD 관리자", "전자결재 관리자", "메일 대용량첨부", "VPN 접속"],
    "DB운영팀": ["운영DB READ", "운영DB WRITE", "운영DB ADMIN", "개발DB WRITE"],
    "개발플랫폼팀": ["형상관리 WRITE", "개발계 배포", "운영계 배포", "개발DB READ"],
    "IT서비스데스크": ["티켓 시스템 관리자", "VPN 접속", "메일 대용량첨부"],
}

SLACK_ID_CHARS = string.ascii_uppercase + string.digits


@dataclass
class _Slot:
    """이름·연락처를 붙이기 전의 사용자 골격."""

    role: UserRole
    user_group: str
    department: str
    it_team: str | None = None
    account_status: AccountStatus = AccountStatus.ACTIVE


def quota(shares: dict[str, float], total: int) -> dict[str, int]:
    """비율을 정수 인원으로 (최대 잔여 배분, 합계 = total). 동률은 shares 순서."""
    raw = {k: total * v / sum(shares.values()) for k, v in shares.items()}
    counts = {k: int(v) for k, v in raw.items()}
    by_remainder = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in by_remainder[: total - sum(counts.values())]:
        counts[k] += 1
    return counts


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _staff_slots(world: World) -> list[_Slot]:
    teams = [t.canonical_name for t in world.teams]
    if MIN_STAFF_PER_TEAM * len(teams) > IT_STAFF_COUNT:
        raise ValueError(f"it_staff {IT_STAFF_COUNT}명으로 팀 {len(teams)}개에 2명씩 배치 불가")
    per_team = {t: MIN_STAFF_PER_TEAM for t in teams}
    for i in range(IT_STAFF_COUNT - MIN_STAFF_PER_TEAM * len(teams)):
        per_team[teams[i % len(teams)]] += 1
    slots = []
    for team, n in per_team.items():
        for i in range(n):
            group = "시스템 관리자" if i == 0 else "정규직"
            slots.append(_Slot("it_staff", group, department=team, it_team=team))
    return slots


def _requester_slots(rng: random.Random, count: int) -> list[_Slot]:
    slots = []
    for group, n in quota(GROUP_SHARES, count).items():
        departments = GROUP_DEPARTMENTS.get(group, BUSINESS_DEPARTMENTS)
        slots.extend(_Slot("requester", group, rng.choice(departments)) for _ in range(n))
    return slots


def _permission_candidates(slot: _Slot, denied: set[tuple[str, str]]) -> list[str]:
    pool = (
        TEAM_PERMISSIONS.get(slot.it_team or "", [])
        + DEPARTMENT_PERMISSIONS.get(slot.department, [])
        + GROUP_PERMISSIONS.get(slot.user_group, [])
    )
    unique = list(dict.fromkeys(pool))
    return [p for p in unique if (slot.user_group, p) not in denied]


def generate_users(
    world: World,
    golds: Sequence[GoldKnowledge],
    as_of: date,
    seed: int = SEED,
    total: int = TOTAL_USERS,
) -> list[MockUser]:
    rng = random.Random(seed)
    fake = Faker("ko_KR")
    fake.seed_instance(seed)

    staff = _staff_slots(world)
    requesters = _requester_slots(rng, total - len(staff))
    for status, share in STATUS_SHARES.items():
        active = [s for s in requesters if s.account_status == AccountStatus.ACTIVE]
        for slot in rng.sample(active, round(total * share)):
            slot.account_status = status

    slots = staff + requesters
    rng.shuffle(slots)
    denied = {(r.user_group, r.permission) for r in active_deny_rules(golds, as_of)}

    names: set[str] = set()
    emails: set[str] = set()
    slack_ids: set[str] = set()
    year_seq: Counter[int] = Counter()

    def unique(generate, seen: set[str]) -> str:
        while (value := generate()) in seen:
            pass
        seen.add(value)
        return value

    users = []
    for i, slot in enumerate(slots, start=1):
        if slot.role == "it_staff":
            hire_from, hire_to = IT_STAFF_HIRE_RANGE
        else:
            hire_from, hire_to = HIRE_RANGES.get(slot.user_group, DEFAULT_HIRE_RANGE)
        hire_date = _random_date(rng, hire_from, hire_to)
        year_seq[hire_date.year] += rng.randint(1, 5)

        candidates = _permission_candidates(slot, denied)
        k = rng.randint(0, min(4, len(candidates)))

        users.append(
            MockUser(
                user_id=f"U{i:04d}",
                employee_no=f"{hire_date.year:04d}{year_seq[hire_date.year]:03d}",
                name=unique(fake.name, names),
                email=unique(lambda: f"{fake.user_name().lower()}@{EMAIL_DOMAIN}", emails),
                phone=f"010-{rng.randint(1000, 9999)}-{rng.randint(0, 9999):04d}",
                slack_user_id=unique(
                    lambda: "S" + "".join(rng.choices(SLACK_ID_CHARS, k=8)), slack_ids
                ),
                department=slot.department,
                user_group=slot.user_group,
                role=slot.role,
                it_team=slot.it_team,
                hire_date=hire_date,
                account_status=slot.account_status,
                granted_permissions=rng.sample(candidates, k),
            )
        )
    return users


def write_users(users: Sequence[MockUser], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for u in users:
            f.write(u.model_dump_json() + "\n")


def summarize(users: Sequence[MockUser]) -> str:
    def block(title: str, counter: Counter[str]) -> list[str]:
        lines = [f"[{title}]"]
        lines += [f"  {k:<12} {v:>4}" for k, v in counter.most_common()]
        return lines

    requesters = [u for u in users if u.role == "requester"]
    lines = [
        f"총 {len(users)}명 (requester {len(requesters)}, it_staff {len(users) - len(requesters)})"
    ]
    lines += block("user_group (전체)", Counter(u.user_group for u in users))
    lines += block("user_group (requester)", Counter(u.user_group for u in requesters))
    lines += block("department", Counter(u.department for u in users))
    lines += block("account_status", Counter(u.account_status.value for u in users))
    lines += block("it_team 담당자 수", Counter(u.it_team for u in users if u.it_team))
    lines += block("권한 개수", Counter(f"{len(u.granted_permissions)}개" for u in users))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mock 사용자 생성 (LLM 호출 없음)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="DENY 정책 유효 판정 기준일 (기본: 오늘)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args(argv)

    world = load_world(WORLD_FILE)
    golds = load_gold_knowledge(GOLD_DIR)
    users = generate_users(world, golds, args.as_of, seed=args.seed)

    errors = validate_users(users, world, golds, args.as_of, expected_count=TOTAL_USERS)
    if errors:
        print(f"검증 실패 ({len(errors)}건) — 저장하지 않습니다", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    write_users(users, args.output)
    print(f"저장: {args.output} (seed={args.seed}, 기준일={args.as_of})")
    print(summarize(users))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
