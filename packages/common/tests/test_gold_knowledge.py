from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from common.schema.gold import GoldKnowledge, ScenarioType, load_gold_knowledge, load_gold_yaml
from common.schema.knowledge import PolicyEffect

GOLD_DIR = Path(__file__).resolve().parents[3] / "data" / "gold" / "knowledge"


@pytest.fixture(scope="module")
def gold() -> list[GoldKnowledge]:
    return load_gold_knowledge(GOLD_DIR)


def test_all_five_files_load_and_validate(gold: list[GoldKnowledge]) -> None:
    assert len(list(GOLD_DIR.glob("*.yaml"))) == 5
    assert [k.id for k in gold] == ["K001", "K002", "K003", "K004", "K005"]


def test_each_scenario_type_appears_once(gold: list[GoldKnowledge]) -> None:
    assert sorted(k.scenario_type for k in gold) == sorted(get_args(ScenarioType))


def test_ids_unique_and_match_filenames(gold: list[GoldKnowledge]) -> None:
    paths = sorted(GOLD_DIR.glob("*.yaml"))

    assert len({k.id for k in gold}) == len(gold)
    assert [p.stem for p in paths] == [k.id for k in gold]


def test_k004_policy_periods_do_not_overlap(gold: list[GoldKnowledge]) -> None:
    k004 = next(k for k in gold if k.id == "K004")
    old, new = sorted(k004.policies, key=lambda p: p.valid_from)

    assert old.effect is PolicyEffect.ALLOW
    assert new.effect is PolicyEffect.REQUIRE_APPROVAL
    assert old.valid_to is not None
    assert old.valid_to < new.valid_from
    assert new.valid_to is None


def _raw(name: str) -> dict:
    return load_gold_yaml((GOLD_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_relation_to_missing_entity_is_rejected() -> None:
    data = _raw("K001")
    data["relations"][1]["target_name"] = "VPN 재설치"

    with pytest.raises(ValidationError, match="entities에 없습니다"):
        GoldKnowledge.model_validate(data)


def test_policy_referencing_missing_group_is_rejected() -> None:
    data = _raw("K003")
    data["policies"][0]["subject_group"] = "인턴"

    with pytest.raises(ValidationError, match="subject_group"):
        GoldKnowledge.model_validate(data)


def test_loader_reports_failing_filename(tmp_path: Path) -> None:
    data = _raw("K001")
    data["entities"][0]["evidence"] = ""
    (tmp_path / "K001.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")

    with pytest.raises(ValueError, match="K001.yaml"):
        load_gold_knowledge(tmp_path)


def test_unquoted_on_relation_is_not_read_as_boolean() -> None:
    # YAML 1.1에서 ON은 불리언 True가 되므로 로더가 이를 막아야 한다.
    text = (GOLD_DIR / "K003.yaml").read_text(encoding="utf-8").replace('"ON"', "ON")
    item = GoldKnowledge.model_validate(load_gold_yaml(text))

    assert item.relations[0].relation_type == "ON"
