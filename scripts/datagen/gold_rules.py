"""gold 지식 검증 규칙 (LLM 호출 없음).

generate_gold.py가 생성 직후·저장 전에 실행한다. 1단계 스키마 검증(GoldKnowledge:
관계 방향·관계 끝점·policies 참조)을 먼저 통과해야 하고, 그 위에 아래를 검사한다.

- world 대조: System·Team·UserGroup·Permission의 canonical_name은 같은 종류의 world
  canonical_name이어야 하고, aliases는 그 world 정의 aliases의 부분집합이어야 한다.
  Permission.system_name·ON 대상은 world on_system, DEPENDS_ON·OWNED_BY는 world 정의와 일치.
- Error.code는 world.error_codes에 있어야 하고, OCCURS_IN 대상이 그 오류의 system이어야 한다.
- 정책 일관성: Policy 엔티티마다 같은 이름의 policies 항목이 있고 유효기간이 같아야 하며,
  policies의 (system_name, level)에 해당하는 Permission 엔티티가 있어야 한다.
- 날짜(valid_from·valid_to)는 DATE_MIN ~ DATE_MAX 범위.
- 유형별 규칙 (check_scenario 참고).
- 중복: 같은 (유형, 핵심 엔티티 조합)의 gold가 이미 있으면 거부.

policy_revision의 기간 규칙: 이전 정책 valid_to = 신규 정책 valid_from 전날
(종료일 포함 규칙, K004와 동일). 사용자 결정 2026-11-23.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

from common.schema.gold import GoldKnowledge
from common.schema.knowledge import EntityType, KnowledgeType, RelationType, SourceType
from common.schema.world import World

DATE_MIN = date(2025, 1, 1)
DATE_MAX = date(2026, 11, 30)

# 핵심 엔티티: 이름이 world에 고정된 종류 + Error. Solution·Policy 이름은 자유 표기라 제외한다.
CORE_TYPES = (
    EntityType.SYSTEM,
    EntityType.ERROR,
    EntityType.PERMISSION,
    EntityType.USER_GROUP,
    EntityType.TEAM,
)

_EFFECTIVE_DATE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)\s*$")

CoreKey = tuple[str, frozenset[tuple[str, str]]]


def core_key(item: GoldKnowledge) -> CoreKey:
    """중복 판정 키: (유형, 핵심 엔티티 (type, 이름) 집합). Error는 code가 있으면 code로."""
    names: set[tuple[str, str]] = set()
    for e in item.entities:
        if e.type not in CORE_TYPES:
            continue
        name = e.code if e.type is EntityType.ERROR and e.code else e.canonical_name
        names.add((str(e.type), name))
    return item.scenario_type, frozenset(names)


def validate_gold(
    item: GoldKnowledge, world: World, existing: Iterable[GoldKnowledge] = ()
) -> list[str]:
    """스키마 검증을 통과한 gold에 대한 추가 규칙 위반 목록. 비어 있으면 통과."""
    errors = [
        *check_world(item, world),
        *check_policy_consistency(item),
        *check_dates(item),
        *check_scenario(item),
    ]
    key = core_key(item)
    for other in existing:
        if other.id != item.id and core_key(other) == key:
            errors.append(
                f"중복: {other.id} ({other.title})와 유형·핵심 엔티티 조합이 같습니다 "
                f"{sorted(n for _, n in key[1])}"
            )
            break
    return errors


# ---------------------------------------------------------------------------
# world 대조
# ---------------------------------------------------------------------------


def check_world(item: GoldKnowledge, world: World) -> list[str]:
    defs = {
        EntityType.SYSTEM: {s.canonical_name: s for s in world.systems},
        EntityType.TEAM: {t.canonical_name: t for t in world.teams},
        EntityType.USER_GROUP: {g.canonical_name: g for g in world.user_groups},
        EntityType.PERMISSION: {p.canonical_name: p for p in world.permissions},
    }
    error_codes = {e.code: e for e in world.error_codes}
    systems = defs[EntityType.SYSTEM]
    errors: list[str] = []

    for e in item.entities:
        if e.type in defs:
            wdef = defs[e.type].get(e.canonical_name)
            if wdef is None:
                ref = world.lookup(e.canonical_name)
                hint = f" (world에서는 {ref.kind} {ref.canonical_name!r})" if ref else ""
                errors.append(f"{e.type} {e.canonical_name!r}: world canonical_name이 아님{hint}")
                continue
            extra = sorted(set(e.aliases) - set(wdef.aliases))
            if extra:
                errors.append(f"{e.type} {e.canonical_name!r}: world aliases에 없는 별칭 {extra}")
            if e.type is EntityType.PERMISSION and e.system_name != wdef.on_system:
                errors.append(
                    f"Permission {e.canonical_name!r}.system_name {e.system_name!r} ≠ "
                    f"world on_system {wdef.on_system!r}"
                )
        elif e.type is EntityType.ERROR and e.code is not None:
            wcode = error_codes.get(e.code)
            if wcode is None:
                errors.append(f"Error {e.canonical_name!r}: code {e.code!r}가 world에 없음")
                continue
            targets = {
                r.target_name
                for r in item.relations
                if r.relation_type is RelationType.OCCURS_IN and r.source_name == e.canonical_name
            }
            if targets != {wcode.system}:
                errors.append(
                    f"Error {e.code}: OCCURS_IN 대상 {sorted(targets)} ≠ world system "
                    f"[{wcode.system!r}]"
                )

    perms = defs[EntityType.PERMISSION]
    for r in item.relations:
        if r.relation_type is RelationType.ON:
            wperm = perms.get(r.source_name)
            if wperm and r.target_name != wperm.on_system:
                errors.append(
                    f"{r.source_name} ON {r.target_name}: world on_system은 {wperm.on_system!r}"
                )
        elif r.relation_type is RelationType.DEPENDS_ON:
            wsys = systems.get(r.source_name)
            if wsys and r.target_name not in wsys.depends_on:
                errors.append(
                    f"{r.source_name} DEPENDS_ON {r.target_name}: world depends_on은 "
                    f"{wsys.depends_on}"
                )
        elif r.relation_type is RelationType.OWNED_BY:
            wsys = systems.get(r.source_name)
            if wsys and r.target_name != wsys.owner_team:
                errors.append(
                    f"{r.source_name} OWNED_BY {r.target_name}: world owner_team은 "
                    f"{wsys.owner_team!r}"
                )
    return errors


# ---------------------------------------------------------------------------
# 정책·날짜
# ---------------------------------------------------------------------------


def check_policy_consistency(item: GoldKnowledge) -> list[str]:
    errors: list[str] = []
    by_name = {p.name: p for p in item.policies}
    for e in _entities(item, EntityType.POLICY):
        p = by_name.get(e.canonical_name)
        if p is None:
            errors.append(f"Policy {e.canonical_name!r}에 대응하는 policies 항목이 없음")
        elif (p.valid_from, p.valid_to) != (e.valid_from, e.valid_to):
            errors.append(f"Policy {e.canonical_name!r}: 엔티티와 policies의 유효기간이 다름")
    perms = {(e.system_name, e.level) for e in _entities(item, EntityType.PERMISSION)}
    for p in item.policies:
        if (p.system_name, p.level) not in perms:
            errors.append(
                f"policies {p.name!r}: ({p.system_name}, {p.level})인 Permission 엔티티가 없음"
            )
    return errors


def check_dates(item: GoldKnowledge) -> list[str]:
    errors: list[str] = []
    dated = [
        (f"{e.type} {e.canonical_name!r}", e) for e in item.entities if hasattr(e, "valid_from")
    ]
    dated += [(f"policies {p.name!r}", p) for p in item.policies]
    for label, obj in dated:
        for field in ("valid_from", "valid_to"):
            value = getattr(obj, field)
            if value is not None and not DATE_MIN <= value <= DATE_MAX:
                errors.append(f"{label}.{field} {value}가 {DATE_MIN}~{DATE_MAX} 범위 밖")
    return errors


# ---------------------------------------------------------------------------
# 유형별 규칙
# ---------------------------------------------------------------------------


def check_scenario(item: GoldKnowledge) -> list[str]:
    return _SCENARIO_RULES[item.scenario_type](item)


def _tacit_fix(item: GoldKnowledge) -> list[str]:
    errors = _require_resolution(item)
    for s in _entities(item, EntityType.SOLUTION):
        if s.knowledge_type is not KnowledgeType.TACIT:
            errors.append(f"tacit_fix: Solution {s.canonical_name!r}.knowledge_type이 tacit이 아님")
    return errors


def _official_procedure(item: GoldKnowledge) -> list[str]:
    errors = _require_resolution(item)
    for s in _entities(item, EntityType.SOLUTION):
        if s.knowledge_type is KnowledgeType.TACIT:
            errors.append(f"official_procedure: Solution {s.canonical_name!r}가 tacit")
    if SourceType.GUIDE not in item.render_as:
        errors.append("official_procedure: render_as에 guide가 없음")
    return errors


def _policy(item: GoldKnowledge) -> list[str]:
    errors = _require_policy_parts(item)
    if _relations(item, RelationType.SUPERSEDES):
        errors.append("policy: SUPERSEDES는 policy_revision 유형에서만 씁니다")
    return errors


def _policy_revision(item: GoldKnowledge) -> list[str]:
    errors = _require_policy_parts(item)
    supersedes = _relations(item, RelationType.SUPERSEDES)
    if not supersedes:
        errors.append("policy_revision: SUPERSEDES 관계가 없음")
    policies = {e.canonical_name: e for e in _entities(item, EntityType.POLICY)}
    for name, e in policies.items():
        m = _EFFECTIVE_DATE.search(name)
        if e.valid_from is None:
            errors.append(f"policy_revision: Policy {name!r}.valid_from이 없음")
        elif m is None or m.group(1) != e.valid_from.isoformat():
            errors.append(
                f"policy_revision: Policy {name!r} 이름 끝에 시행일 ({e.valid_from})이 없음"
            )
    for r in supersedes:
        new, old = policies.get(r.source_name), policies.get(r.target_name)
        if new is None or old is None or new.valid_from is None:
            continue
        expected = new.valid_from - timedelta(days=1)
        if old.valid_to != expected:
            errors.append(
                f"policy_revision: 이전 정책 {old.canonical_name!r}.valid_to {old.valid_to} ≠ "
                f"신규 정책 시행일 전날 {expected}"
            )
    return errors


def _multihop(item: GoldKnowledge) -> list[str]:
    if longest_path(item) < 2:
        return ["multihop: 관계 방향을 따라 2홉 이상 이어지는 경로가 없음"]
    return []


def _require_resolution(item: GoldKnowledge) -> list[str]:
    errors: list[str] = []
    if not _entities(item, EntityType.SOLUTION):
        errors.append(f"{item.scenario_type}: Solution 엔티티가 없음")
    if not _relations(item, RelationType.RESOLVED_BY):
        errors.append(f"{item.scenario_type}: Error RESOLVED_BY Solution 관계가 없음")
    return errors


def _require_policy_parts(item: GoldKnowledge) -> list[str]:
    st = item.scenario_type
    errors: list[str] = []
    if not item.policies:
        errors.append(f"{st}: policies(효과 ALLOW/DENY/REQUIRE_APPROVAL)가 비어 있음")
    if not _entities(item, EntityType.USER_GROUP):
        errors.append(f"{st}: UserGroup 엔티티가 없음")
    if not _entities(item, EntityType.PERMISSION):
        errors.append(f"{st}: Permission 엔티티가 없음")
    if not _relations(item, RelationType.GOVERNS):
        errors.append(f"{st}: Policy GOVERNS Permission 관계가 없음")
    if not _relations(item, RelationType.APPLIES_TO):
        errors.append(f"{st}: Policy APPLIES_TO UserGroup 관계가 없음")
    return errors


_SCENARIO_RULES = {
    "tacit_fix": _tacit_fix,
    "official_procedure": _official_procedure,
    "policy": _policy,
    "policy_revision": _policy_revision,
    "multihop": _multihop,
}


# ---------------------------------------------------------------------------
# 그래프 유틸
# ---------------------------------------------------------------------------


def longest_path(item: GoldKnowledge) -> int:
    """관계 방향을 따라가는 가장 긴 단순 경로의 홉 수."""
    graph: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in item.relations:
        src = (str(r.source_type), r.source_name)
        graph.setdefault(src, []).append((str(r.target_type), r.target_name))

    def walk(node: tuple[str, str], seen: frozenset[tuple[str, str]]) -> int:
        return max(
            (1 + walk(nxt, seen | {nxt}) for nxt in graph.get(node, []) if nxt not in seen),
            default=0,
        )

    return max((walk(n, frozenset({n})) for n in graph), default=0)


def _entities(item: GoldKnowledge, entity_type: EntityType) -> list:
    return [e for e in item.entities if e.type is entity_type]


def _relations(item: GoldKnowledge, relation_type: RelationType) -> list:
    return [r for r in item.relations if r.relation_type is relation_type]


def system_mentions(items: Sequence[GoldKnowledge]) -> dict[str, int]:
    """System 엔티티 canonical_name별 등장 gold 수."""
    counts: dict[str, int] = {}
    for item in items:
        for name in {e.canonical_name for e in _entities(item, EntityType.SYSTEM)}:
            counts[name] = counts.get(name, 0) + 1
    return counts
