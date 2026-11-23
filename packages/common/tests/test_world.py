import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from common.schema.gold import load_gold_yaml
from common.schema.world import RiskLevel, World, load_world

WORLD_FILE = Path(__file__).resolve().parents[3] / "data" / "world" / "world.yaml"


def _minimal() -> dict[str, Any]:
    return {
        "teams": [{"canonical_name": "보안운영팀", "aliases": ["보안팀"], "description": "보안"}],
        "user_groups": [
            {"canonical_name": "외주", "aliases": ["외주 인력"], "description": "외주"}
        ],
        "systems": [
            {
                "canonical_name": "SSO",
                "aliases": ["통합 로그인"],
                "owner_team": "보안운영팀",
                "depends_on": [],
                "description": "통합 인증",
            },
            {
                "canonical_name": "MFA",
                "aliases": ["2차 인증"],
                "owner_team": "보안운영팀",
                "depends_on": ["SSO"],
                "description": "다중 인증",
            },
            {
                "canonical_name": "VPN",
                "aliases": ["사내 VPN"],
                "owner_team": "보안운영팀",
                "depends_on": ["MFA"],
                "description": "원격접속",
            },
        ],
        "permissions": [
            {
                "canonical_name": "VPN 접속",
                "aliases": [],
                "on_system": "VPN",
                "risk_level": "MEDIUM",
            }
        ],
        "error_codes": [
            {"code": "E-102", "system": "VPN", "message": "연결 끊김", "description": "터널 실패"}
        ],
    }


# -- 정상 로드 ---------------------------------------------------------------


def test_world_yaml_loads_with_expected_size() -> None:
    world = load_world(WORLD_FILE)

    assert 14 <= len(world.systems) <= 17
    assert len(world.teams) == 6
    assert len(world.user_groups) == 5
    assert 18 <= len(world.permissions) <= 22
    assert 28 <= len(world.error_codes) <= 35
    assert all(1 <= len(s.aliases) <= 3 for s in world.systems)
    assert {p.risk_level for p in world.permissions} == set(RiskLevel)

    # 2~3단계(시스템 3~4개) 의존 체인이 서로 다른 출발점에서 3개 이상
    starts = {
        s.canonical_name
        for s in world.systems
        if any(len(chain) >= 3 for chain in world.dependency_chain(s.canonical_name))
    }
    assert len(starts) >= 3
    assert ["VPN", "MFA", "SSO", "AD 계정"] in world.dependency_chain("VPN")


def test_minimal_world_is_valid_and_lookup_resolves_aliases() -> None:
    world = World.model_validate(_minimal())

    ref = world.lookup("사내vpn")  # 대소문자·공백 무시
    assert ref is not None
    assert (ref.kind, ref.canonical_name, ref.is_alias) == ("system", "VPN", True)
    assert world.lookup("e-102").kind == "error_code"
    assert world.lookup("없는 이름") is None


# -- 잘못된 참조 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("systems", 0, "owner_team"), "없는팀", "owner_team"),
        (("systems", 1, "depends_on"), ["없는시스템"], "depends_on"),
        (("permissions", 0, "on_system"), "사내 VPN", "on_system"),  # 별칭은 참조 불가
        (("error_codes", 0, "system"), "없는시스템", "system"),
    ],
)
def test_unknown_reference_is_rejected(path: tuple, value: Any, message: str) -> None:
    data = copy.deepcopy(_minimal())
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        World.model_validate(data)


# -- 중복 alias ---------------------------------------------------------------


def test_duplicate_alias_across_kinds_is_rejected() -> None:
    data = _minimal()
    data["user_groups"][0]["aliases"].append("보안 팀")  # 팀 별칭 "보안팀"과 공백만 다름

    with pytest.raises(ValidationError, match="이름 중복"):
        World.model_validate(data)


def test_alias_equal_to_canonical_name_is_rejected() -> None:
    data = _minimal()
    data["systems"][0]["aliases"].append("vpn")

    with pytest.raises(ValidationError, match="이름 중복"):
        World.model_validate(data)


# -- 순환 의존 ----------------------------------------------------------------


def test_dependency_cycle_is_rejected() -> None:
    data = _minimal()
    data["systems"][0]["depends_on"] = ["VPN"]  # SSO -> VPN -> MFA -> SSO

    with pytest.raises(ValidationError, match="순환"):
        World.model_validate(data)


def test_self_dependency_is_rejected() -> None:
    data = _minimal()
    data["systems"][0]["depends_on"] = ["SSO"]

    with pytest.raises(ValidationError, match="순환"):
        World.model_validate(data)


def test_load_world_error_includes_filename(tmp_path: Path) -> None:
    data = load_gold_yaml(WORLD_FILE.read_text(encoding="utf-8"))
    data["systems"][0]["depends_on"] = ["VPN"]  # AD 계정 -> VPN -> MFA -> SSO -> AD 계정
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"(?s)broken\.yaml: .*순환"):
        load_world(path)
