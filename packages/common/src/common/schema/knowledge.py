"""지식 추출 스키마.

LLM이 문서 청크 1개를 읽고 엔티티·관계·정책을 뽑을 때 쓰는 구조화 출력 양식.
Source 노드와 EXTRACTED_FROM 관계는 적재 파이프라인이 붙이므로 여기서 다루지 않는다.

Azure OpenAI 구조화 출력(strict) 호환을 위해:
- 모든 모델은 extra="forbid"
- 자유 형식(dict[str, Any]) 필드 없음
- 선택 값은 기본값 대신 nullable로 두고 필드는 전부 required
- 값 제약(공백 금지 등)은 JSON 스키마 키워드가 아니라 validator로 검사한다
"""

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    SYSTEM = "System"
    ERROR = "Error"
    SOLUTION = "Solution"
    PERMISSION = "Permission"
    USER_GROUP = "UserGroup"
    POLICY = "Policy"
    TEAM = "Team"


class RelationType(StrEnum):
    OCCURS_IN = "OCCURS_IN"
    RESOLVED_BY = "RESOLVED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    ON = "ON"
    GOVERNS = "GOVERNS"
    APPLIES_TO = "APPLIES_TO"
    SUPERSEDES = "SUPERSEDES"
    OWNED_BY = "OWNED_BY"


class KnowledgeType(StrEnum):
    OFFICIAL = "official"
    TACIT = "tacit"


class PermissionLevel(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    ADMIN = "ADMIN"


class PolicyEffect(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class UserGroupKind(StrEnum):
    EMPLOYMENT = "employment"
    ROLE = "role"


class SourceType(StrEnum):
    """청크 메타데이터용 원천 유형."""

    GUIDE = "guide"
    SLACK = "slack"
    TICKET = "ticket"
    MINUTES = "minutes"


# (source_type, relation_type, target_type) 허용 조합
ALLOWED_RELATIONS: frozenset[tuple[EntityType, RelationType, EntityType]] = frozenset(
    {
        (EntityType.ERROR, RelationType.OCCURS_IN, EntityType.SYSTEM),
        (EntityType.ERROR, RelationType.RESOLVED_BY, EntityType.SOLUTION),
        (EntityType.SYSTEM, RelationType.DEPENDS_ON, EntityType.SYSTEM),
        (EntityType.PERMISSION, RelationType.ON, EntityType.SYSTEM),
        (EntityType.POLICY, RelationType.GOVERNS, EntityType.PERMISSION),
        (EntityType.POLICY, RelationType.APPLIES_TO, EntityType.USER_GROUP),
        (EntityType.POLICY, RelationType.SUPERSEDES, EntityType.POLICY),
        (EntityType.SYSTEM, RelationType.OWNED_BY, EntityType.TEAM),
    }
)

# ---------------------------------------------------------------------------
# 공용 타입
# ---------------------------------------------------------------------------


def _strip_non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("빈 값은 허용되지 않습니다")
    return stripped


# 앞뒤 공백을 제거하고 빈 값을 거부하는 문자열.
# StringConstraints(minLength) 대신 validator를 써서 JSON 스키마를 strict 호환으로 유지한다.
NonBlankStr = Annotated[str, AfterValidator(_strip_non_blank)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_validity_period(valid_from: date | None, valid_to: date | None) -> None:
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ValueError("valid_from은 valid_to보다 늦을 수 없습니다")


# ---------------------------------------------------------------------------
# 엔티티
# ---------------------------------------------------------------------------


class _EntityBase(_StrictModel):
    canonical_name: NonBlankStr = Field(description="정규화된 대표 이름")
    aliases: list[NonBlankStr] = Field(description="원문에 등장한 다른 표기")
    evidence: NonBlankStr = Field(description="근거가 되는 원문 인용")


class SystemEntity(_EntityBase):
    type: Literal[EntityType.SYSTEM]
    description: str | None


class ErrorEntity(_EntityBase):
    type: Literal[EntityType.ERROR]
    code: str | None = Field(description="오류 코드 (예: E-102). 없으면 null")
    symptom: NonBlankStr


class SolutionEntity(_EntityBase):
    type: Literal[EntityType.SOLUTION]
    steps: list[NonBlankStr]
    knowledge_type: KnowledgeType
    valid_from: date | None
    valid_to: date | None

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        _check_validity_period(self.valid_from, self.valid_to)
        return self


class PermissionEntity(_EntityBase):
    type: Literal[EntityType.PERMISSION]
    system_name: NonBlankStr
    level: PermissionLevel


class UserGroupEntity(_EntityBase):
    type: Literal[EntityType.USER_GROUP]
    kind: UserGroupKind


class PolicyEntity(_EntityBase):
    """정책 노드. 정책 세부 내용은 ExtractedPolicy로 따로 추출한다."""

    type: Literal[EntityType.POLICY]
    valid_from: date | None
    valid_to: date | None

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        _check_validity_period(self.valid_from, self.valid_to)
        return self


class TeamEntity(_EntityBase):
    type: Literal[EntityType.TEAM]
    channel: str | None = Field(description="팀 Slack 채널. 없으면 null")


ExtractedEntity = Annotated[
    SystemEntity
    | ErrorEntity
    | SolutionEntity
    | PermissionEntity
    | UserGroupEntity
    | PolicyEntity
    | TeamEntity,
    Field(discriminator="type"),
]

# ---------------------------------------------------------------------------
# 관계·정책
# ---------------------------------------------------------------------------


class ExtractedRelation(_StrictModel):
    source_type: EntityType
    source_name: NonBlankStr
    relation_type: RelationType
    target_type: EntityType
    target_name: NonBlankStr
    evidence: NonBlankStr

    @model_validator(mode="after")
    def _check_allowed(self) -> Self:
        triple = (self.source_type, self.relation_type, self.target_type)
        if triple not in ALLOWED_RELATIONS:
            raise ValueError(
                f"허용되지 않은 관계 조합: "
                f"{self.source_type}-{self.relation_type}->{self.target_type}"
            )
        return self


class ExtractedPolicy(_StrictModel):
    name: NonBlankStr
    subject_group: NonBlankStr = Field(description="대상 UserGroup의 canonical_name")
    system_name: NonBlankStr
    level: PermissionLevel
    effect: PolicyEffect
    conditions: str | None
    valid_from: date | None
    valid_to: date | None
    evidence: NonBlankStr

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        _check_validity_period(self.valid_from, self.valid_to)
        return self


class ExtractionResult(_StrictModel):
    chunk_id: NonBlankStr
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    policies: list[ExtractedPolicy]

    @model_validator(mode="after")
    def _check_relation_endpoints(self) -> Self:
        check_relation_endpoints(self.entities, self.relations)
        return self


def check_relation_endpoints(
    entities: Sequence[_EntityBase], relations: Sequence[ExtractedRelation]
) -> None:
    """relations가 가리키는 (type, canonical_name)이 entities에 있는지 확인한다."""
    known = {(e.type, e.canonical_name) for e in entities}
    for i, rel in enumerate(relations):
        for role, key in (
            ("source", (rel.source_type, rel.source_name)),
            ("target", (rel.target_type, rel.target_name)),
        ):
            if key not in known:
                raise ValueError(
                    f"relations[{i}].{role} ({key[0]}, {key[1]!r})가 entities에 없습니다"
                )


# ---------------------------------------------------------------------------
# JSON 스키마
# ---------------------------------------------------------------------------


class _StrictJsonSchema(GenerateJsonSchema):
    """구조화 출력(strict)이 지원하지 않는 키워드를 지원 키워드로 바꾼다.

    - discriminated union: oneOf + discriminator → anyOf
    - 단일 Literal: const → enum
    """

    def tagged_union_schema(self, schema: core_schema.TaggedUnionSchema) -> JsonSchemaValue:
        json_schema = super().tagged_union_schema(schema)
        json_schema.pop("discriminator", None)
        if "oneOf" in json_schema:
            json_schema["anyOf"] = json_schema.pop("oneOf")
        return json_schema

    def literal_schema(self, schema: core_schema.LiteralSchema) -> JsonSchemaValue:
        json_schema = super().literal_schema(schema)
        if "const" in json_schema:
            json_schema["enum"] = [json_schema.pop("const")]
        return json_schema


def extraction_json_schema() -> dict[str, Any]:
    """ExtractionResult의 JSON 스키마 (Azure OpenAI 구조화 출력용)."""
    return ExtractionResult.model_json_schema(schema_generator=_StrictJsonSchema)
