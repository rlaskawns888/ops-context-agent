"""세계관(world.yaml) 검증 + gold 지식의 엔티티 이름 대조.

실행:
    uv run python scripts/validate_world.py

1. data/world/world.yaml을 로드한다 (참조·중복·순환 검증은 World 모델이 수행).
2. data/gold/knowledge/*.yaml의 엔티티 이름이 world에 canonical_name으로 있는지 확인하고
   없는 이름을 출력한다. gold 파일은 수정하지 않는다.
   - System·Team·UserGroup·Permission: canonical_name을 같은 종류의 world 정의와 대조
   - Error: code가 있으면 world error_codes와 대조, code가 없으면 대상 아님(건너뜀)
   - Solution·Policy: 세계관 범위 밖이라 건너뜀

종료 코드: world 검증 실패 또는 없는 이름이 있으면 1, 모두 통과하면 0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.knowledge import EntityType
from common.schema.world import World, load_world

ROOT = Path(__file__).resolve().parents[1]
WORLD_FILE = ROOT / "data" / "world" / "world.yaml"
GOLD_DIR = ROOT / "data" / "gold" / "knowledge"

_KIND_BY_TYPE = {
    EntityType.SYSTEM: "system",
    EntityType.TEAM: "team",
    EntityType.USER_GROUP: "user_group",
    EntityType.PERMISSION: "permission",
}


@dataclass(frozen=True)
class Issue:
    gold_id: str
    entity_type: str
    name: str
    reason: str


def find_missing(world: World, gold: list[GoldKnowledge]) -> list[Issue]:
    """gold 엔티티 중 world에 canonical_name(또는 오류 code)으로 정의되지 않은 것."""
    issues: list[Issue] = []
    for item in gold:
        for entity in item.entities:
            if entity.type is EntityType.ERROR:
                if entity.code is None:
                    continue
                name, kind = entity.code, "error_code"
            elif entity.type in _KIND_BY_TYPE:
                name, kind = entity.canonical_name, _KIND_BY_TYPE[entity.type]
            else:
                continue

            ref = world.lookup(name)
            if ref is None:
                reason = "world에 없음"
            elif ref.kind != kind:
                reason = f"world에서는 {ref.kind} {ref.canonical_name!r}의 이름"
            elif ref.is_alias or ref.canonical_name != name:
                reason = f"world canonical_name은 {ref.canonical_name!r}"
            else:
                continue
            issues.append(Issue(item.id, str(entity.type), name, reason))
    return issues


def main() -> int:
    try:
        world = load_world(WORLD_FILE)
    except ValueError as exc:
        print(f"[FAIL] world 검증 실패\n{exc}")
        return 1
    print(
        f"[OK] world: systems {len(world.systems)}, teams {len(world.teams)}, "
        f"user_groups {len(world.user_groups)}, permissions {len(world.permissions)}, "
        f"error_codes {len(world.error_codes)}"
    )

    gold = load_gold_knowledge(GOLD_DIR)
    issues = find_missing(world, gold)
    if not issues:
        print(f"[OK] gold {len(gold)}건의 엔티티 이름이 모두 world에 있습니다")
        return 0

    print(f"[FAIL] world에 없는 gold 엔티티 이름 {len(issues)}개")
    for issue in issues:
        print(f"  {issue.gold_id}  {issue.entity_type:<10} {issue.name!r}: {issue.reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
