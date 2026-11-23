"""validate_world.py: gold 엔티티 이름 대조 로직."""

import importlib.util
import sys
from pathlib import Path

from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.world import load_world

_SCRIPT = Path(__file__).resolve().parents[1] / "validate_world.py"
_spec = importlib.util.spec_from_file_location("validate_world", _SCRIPT)
validate_world = importlib.util.module_from_spec(_spec)
sys.modules["validate_world"] = validate_world  # dataclass가 모듈을 조회하므로 등록 필요
_spec.loader.exec_module(validate_world)


def test_current_gold_names_all_exist_in_world() -> None:
    world = load_world(validate_world.WORLD_FILE)
    gold = load_gold_knowledge(validate_world.GOLD_DIR)

    assert validate_world.find_missing(world, gold) == []
    assert validate_world.main() == 0


def test_missing_alias_and_wrong_kind_names_are_reported() -> None:
    world = load_world(validate_world.WORLD_FILE)
    item = GoldKnowledge.model_validate(
        {
            "id": "K999",
            "title": "테스트",
            "scenario_type": "tacit_fix",
            "render_as": ["slack"],
            "entities": [
                _system("사내 VPN"),  # world에서는 별칭
                _system("없는 시스템"),
                _system("보안운영팀"),  # world에서는 팀
                _system("VPN"),  # 정상
                {
                    "type": "Error",
                    "canonical_name": "E-999",
                    "aliases": [],
                    "evidence": "e",
                    "code": "E-999",
                    "symptom": "s",
                },
                {  # code 없는 오류는 대상 아님
                    "type": "Error",
                    "canonical_name": "계정 잠금",
                    "aliases": [],
                    "evidence": "e",
                    "code": None,
                    "symptom": "s",
                },
            ],
            "relations": [],
            "policies": [],
        }
    )

    issues = validate_world.find_missing(world, [item])

    assert [(i.name, i.reason) for i in issues] == [
        ("사내 VPN", "world canonical_name은 'VPN'"),
        ("없는 시스템", "world에 없음"),
        ("보안운영팀", "world에서는 team '보안운영팀'의 이름"),
        ("E-999", "world에 없음"),
    ]


def _system(name: str) -> dict:
    return {
        "type": "System",
        "canonical_name": name,
        "aliases": [],
        "evidence": "e",
        "description": None,
    }
