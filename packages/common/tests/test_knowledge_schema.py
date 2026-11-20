from typing import Any

import pytest
from pydantic import ValidationError

from common.schema.knowledge import (
    EntityType,
    ErrorEntity,
    ExtractedPolicy,
    ExtractedRelation,
    ExtractionResult,
    KnowledgeType,
    PolicyEffect,
    RelationType,
    SolutionEntity,
    extraction_json_schema,
)

VPN_EVIDENCE = "VPN 접속할 때 E-102 뜨면 PC 재부팅하면 됩니다"


def _vpn_payload() -> dict[str, Any]:
    """Slack 예시: VPN 접속 시 E-102 오류 → PC 재부팅으로 해결(tacit)."""
    return {
        "chunk_id": "slack-2026-11-01-001",
        "entities": [
            {
                "type": "System",
                "canonical_name": " VPN ",
                "aliases": ["사내 VPN"],
                "evidence": VPN_EVIDENCE,
                "description": None,
            },
            {
                "type": "Error",
                "canonical_name": "E-102",
                "aliases": [],
                "evidence": VPN_EVIDENCE,
                "code": "E-102",
                "symptom": "VPN 접속 실패",
            },
            {
                "type": "Solution",
                "canonical_name": "PC 재부팅",
                "aliases": ["재부팅"],
                "evidence": VPN_EVIDENCE,
                "steps": ["PC를 재부팅한다", "VPN에 다시 접속한다"],
                "knowledge_type": "tacit",
                "valid_from": None,
                "valid_to": None,
            },
        ],
        "relations": [
            {
                "source_type": "Error",
                "source_name": "E-102",
                "relation_type": "OCCURS_IN",
                "target_type": "System",
                "target_name": "VPN",
                "evidence": VPN_EVIDENCE,
            },
            {
                "source_type": "Error",
                "source_name": "E-102",
                "relation_type": "RESOLVED_BY",
                "target_type": "Solution",
                "target_name": "PC 재부팅",
                "evidence": VPN_EVIDENCE,
            },
        ],
        "policies": [],
    }


def test_vpn_error_solution_example_passes() -> None:
    result = ExtractionResult.model_validate(_vpn_payload())

    system, error, solution = result.entities
    assert system.canonical_name == "VPN"  # 앞뒤 공백 제거
    assert isinstance(error, ErrorEntity)
    assert error.code == "E-102"
    assert isinstance(solution, SolutionEntity)
    assert solution.knowledge_type is KnowledgeType.TACIT
    assert [r.relation_type for r in result.relations] == [
        RelationType.OCCURS_IN,
        RelationType.RESOLVED_BY,
    ]


@pytest.mark.parametrize("evidence", ["", "   "])
def test_blank_evidence_fails(evidence: str) -> None:
    payload = _vpn_payload()
    payload["entities"][0]["evidence"] = evidence

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_blank_relation_evidence_fails() -> None:
    payload = _vpn_payload()
    payload["relations"][0]["evidence"] = ""

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_disallowed_relation_combination_fails() -> None:
    with pytest.raises(ValidationError, match="허용되지 않은 관계 조합"):
        ExtractedRelation(
            source_type=EntityType.SOLUTION,
            source_name="PC 재부팅",
            relation_type=RelationType.OCCURS_IN,
            target_type=EntityType.SYSTEM,
            target_name="VPN",
            evidence=VPN_EVIDENCE,
        )


def test_relation_referencing_unknown_entity_fails() -> None:
    payload = _vpn_payload()
    payload["relations"][1]["target_name"] = "VPN 클라이언트 재설치"

    with pytest.raises(ValidationError, match="entities에 없습니다"):
        ExtractionResult.model_validate(payload)


def test_relation_with_mismatched_entity_type_fails() -> None:
    # 이름은 있지만 타입이 다르면 다른 엔티티로 본다.
    payload = _vpn_payload()
    payload["entities"][1]["type"] = "System"
    payload["entities"][1].pop("code")
    payload["entities"][1].pop("symptom")
    payload["entities"][1]["description"] = None

    with pytest.raises(ValidationError, match="entities에 없습니다"):
        ExtractionResult.model_validate(payload)


def test_outsourcing_prod_db_write_deny_policy_passes() -> None:
    evidence = "외주 인력은 운영DB에 쓰기 권한을 가질 수 없다"
    payload = {
        "chunk_id": "guide-security-003",
        "entities": [
            {
                "type": "UserGroup",
                "canonical_name": "외주",
                "aliases": ["외주 인력"],
                "evidence": evidence,
                "kind": "employment",
            },
            {
                "type": "System",
                "canonical_name": "운영DB",
                "aliases": [],
                "evidence": evidence,
                "description": None,
            },
            {
                "type": "Permission",
                "canonical_name": "운영DB WRITE",
                "aliases": [],
                "evidence": evidence,
                "system_name": "운영DB",
                "level": "WRITE",
            },
            {
                "type": "Policy",
                "canonical_name": "외주 운영DB 쓰기 금지",
                "aliases": [],
                "evidence": evidence,
                "valid_from": "2026-01-01",
                "valid_to": None,
            },
        ],
        "relations": [
            {
                "source_type": "Permission",
                "source_name": "운영DB WRITE",
                "relation_type": "ON",
                "target_type": "System",
                "target_name": "운영DB",
                "evidence": evidence,
            },
            {
                "source_type": "Policy",
                "source_name": "외주 운영DB 쓰기 금지",
                "relation_type": "GOVERNS",
                "target_type": "Permission",
                "target_name": "운영DB WRITE",
                "evidence": evidence,
            },
            {
                "source_type": "Policy",
                "source_name": "외주 운영DB 쓰기 금지",
                "relation_type": "APPLIES_TO",
                "target_type": "UserGroup",
                "target_name": "외주",
                "evidence": evidence,
            },
        ],
        "policies": [
            {
                "name": "외주 운영DB 쓰기 금지",
                "subject_group": "외주",
                "system_name": "운영DB",
                "level": "WRITE",
                "effect": "DENY",
                "conditions": None,
                "valid_from": "2026-01-01",
                "valid_to": None,
                "evidence": evidence,
            }
        ],
    }

    result = ExtractionResult.model_validate(payload)

    (policy,) = result.policies
    assert policy.effect is PolicyEffect.DENY
    assert policy.subject_group == "외주"


def test_policy_with_inverted_validity_period_fails() -> None:
    with pytest.raises(ValidationError, match="valid_from"):
        ExtractedPolicy(
            name="p",
            subject_group="외주",
            system_name="운영DB",
            level="WRITE",
            effect="DENY",
            conditions=None,
            valid_from="2026-12-31",
            valid_to="2026-01-01",
            evidence="근거",
        )


def test_extra_field_is_forbidden() -> None:
    payload = _vpn_payload()
    payload["entities"][0]["owner"] = "인프라팀"

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_missing_nullable_field_fails() -> None:
    # 선택 값도 기본값 없이 required: null을 명시해야 한다.
    payload = _vpn_payload()
    payload["entities"][0].pop("description")

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def _iter_object_schemas(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_iter_object_schemas(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_iter_object_schemas(value))
    return found


def _collect_keys(node: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            # properties 아래의 키는 필드 이름이므로 제외하고 그 값만 본다.
            if key == "properties":
                for prop in value.values():
                    keys |= _collect_keys(prop)
            else:
                keys |= _collect_keys(value)
    elif isinstance(node, list):
        for value in node:
            keys |= _collect_keys(value)
    return keys


def test_extraction_json_schema_is_strict_compatible() -> None:
    schema = extraction_json_schema()

    assert schema["additionalProperties"] is False
    objects = _iter_object_schemas(schema)
    assert len(objects) >= 10  # 최상위 + 엔티티 7종 + 관계 + 정책
    for obj in objects:
        assert obj["additionalProperties"] is False, obj.get("title")
        assert set(obj["required"]) == set(obj["properties"]), obj.get("title")

    unsupported = {"oneOf", "discriminator", "default", "const", "minLength"}
    assert not (_collect_keys(schema) & unsupported)
