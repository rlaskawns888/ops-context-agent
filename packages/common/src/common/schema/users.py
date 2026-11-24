"""mock 사용자(MockUser).

Slack 스레드·티켓 작성자, Agent 업무 처리(계정 잠금 해제·권한 신청) 대상이 되는 가상 사용자.
scripts/datagen/generate_users.py가 고정 seed로 생성해 data/generated/users.jsonl에 저장하고,
5단계에서 ops-api DB 시드가 이 파일을 읽어 적재한다.

모델 자체 검증 (MockUser)
- 형식: user_id U0001~, 사번 7자리(앞 4자리 = 입사연도), 가상 도메인 메일, 010 번호, Slack ID
- role이 it_staff이면 it_team 필수, requester이면 it_team은 null
- granted_permissions는 0~4개, 중복 없음

세계관·정책 대조 (validate_users, 저장 전 실행)
- user_group·it_team·granted_permissions가 world canonical_name이어야 한다
- 기준일(as_of)에 유효한 gold DENY 정책 위반 금지: 정책 대상 UserGroup 사용자가
  그 정책이 GOVERNS하는 Permission을 보유하면 실패. valid_to는 마지막 날 포함.
- user_id·사번·이름·이메일·slack_user_id 중복 금지, 인원 수 일치
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.schema.gold import GoldKnowledge
from common.schema.knowledge import EntityType, NonBlankStr, PolicyEffect, RelationType
from common.schema.world import World

MAX_PERMISSIONS = 4

UserRole = Literal["requester", "it_staff"]


class AccountStatus(StrEnum):
    ACTIVE = "ACTIVE"
    LOCKED = "LOCKED"
    DISABLED = "DISABLED"


class MockUser(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=r"^U\d{4}$")
    employee_no: str = Field(pattern=r"^\d{7}$", description="입사연도 4자리 + 일련번호 3자리")
    name: NonBlankStr
    email: str = Field(pattern=r"^[a-z0-9._-]+@opsctx\.example$")
    phone: str = Field(pattern=r"^010-\d{4}-\d{4}$")
    slack_user_id: str = Field(pattern=r"^S[A-Z0-9]{8}$")
    department: NonBlankStr
    user_group: NonBlankStr = Field(description="world user_groups canonical_name")
    role: UserRole
    it_team: NonBlankStr | None = Field(description="it_staff면 world teams canonical_name")
    hire_date: date
    account_status: AccountStatus
    granted_permissions: list[NonBlankStr] = Field(max_length=MAX_PERMISSIONS)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.role == "it_staff") != (self.it_team is not None):
            raise ValueError(
                "it_staff는 it_team이 필요하고, requester는 it_team이 null이어야 합니다"
            )
        if self.employee_no[:4] != f"{self.hire_date.year:04d}":
            raise ValueError(
                f"사번 {self.employee_no!r}의 앞 4자리가 입사연도 {self.hire_date.year}와 다릅니다"
            )
        if len(set(self.granted_permissions)) != len(self.granted_permissions):
            raise ValueError("granted_permissions에 중복 값이 있습니다")
        return self


# -- 정책 ---------------------------------------------------------------------


@dataclass(frozen=True)
class DenyRule:
    """기준일에 유효한 DENY 정책 1건이 금지하는 (UserGroup, Permission) 조합."""

    gold_id: str
    policy: str
    user_group: str
    permission: str


def is_active(valid_from: date | None, valid_to: date | None, as_of: date) -> bool:
    """as_of에 유효한지. None은 제한 없음, valid_to는 마지막 날 포함."""
    return (valid_from is None or valid_from <= as_of) and (valid_to is None or as_of <= valid_to)


def active_deny_rules(golds: Iterable[GoldKnowledge], as_of: date) -> list[DenyRule]:
    """as_of에 유효한 DENY 정책이 금지하는 (UserGroup, Permission) 목록.

    정책 → 권한은 같은 gold의 Policy -GOVERNS-> Permission 관계로 찾는다.
    (policies의 system_name·level만으로는 같은 시스템의 여러 권한을 구분할 수 없다)
    """
    rules: list[DenyRule] = []
    for gold in golds:
        for policy in gold.policies:
            if policy.effect != PolicyEffect.DENY:
                continue
            if not is_active(policy.valid_from, policy.valid_to, as_of):
                continue
            permissions = [
                r.target_name
                for r in gold.relations
                if r.relation_type == RelationType.GOVERNS
                and r.source_type == EntityType.POLICY
                and r.source_name == policy.name
                and r.target_type == EntityType.PERMISSION
            ]
            if not permissions:
                raise ValueError(
                    f"{gold.id}: DENY 정책 {policy.name!r}이 GOVERNS하는 Permission이 없습니다"
                )
            rules.extend(
                DenyRule(gold.id, policy.name, policy.subject_group, perm) for perm in permissions
            )
    return rules


# -- 검증 ---------------------------------------------------------------------


def validate_users(
    users: Sequence[MockUser],
    world: World,
    golds: Iterable[GoldKnowledge],
    as_of: date,
    expected_count: int | None = None,
) -> list[str]:
    """세계관·정책·고유값 검증. 위반 사유 목록을 돌려준다 (비어 있으면 통과)."""
    errors: list[str] = []

    if expected_count is not None and len(users) != expected_count:
        errors.append(f"사용자 수 {len(users)}명 (기대 {expected_count}명)")

    for field in ("user_id", "employee_no", "name", "email", "slack_user_id"):
        counts = Counter(getattr(u, field) for u in users)
        dupes = sorted(value for value, n in counts.items() if n > 1)
        if dupes:
            errors.append(f"{field} 중복: {dupes}")

    groups = {g.canonical_name for g in world.user_groups}
    teams = {t.canonical_name for t in world.teams}
    permissions = {p.canonical_name for p in world.permissions}
    denied: dict[tuple[str, str], DenyRule] = {
        (r.user_group, r.permission): r for r in active_deny_rules(golds, as_of)
    }

    for u in users:
        if u.user_group not in groups:
            errors.append(f"{u.user_id}: user_group {u.user_group!r}가 world에 없습니다")
        if u.it_team is not None and u.it_team not in teams:
            errors.append(f"{u.user_id}: it_team {u.it_team!r}가 world에 없습니다")
        for perm in u.granted_permissions:
            if perm not in permissions:
                errors.append(f"{u.user_id}: 권한 {perm!r}가 world에 없습니다")
            rule = denied.get((u.user_group, perm))
            if rule is not None:
                errors.append(
                    f"{u.user_id}: {u.user_group} 사용자가 {perm!r} 보유 — "
                    f"{rule.gold_id} DENY 정책 {rule.policy!r} 위반"
                )
    return errors
