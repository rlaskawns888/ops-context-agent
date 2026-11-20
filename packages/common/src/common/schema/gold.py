"""정답 지식(gold knowledge).

YAML 파일 1개 = GoldKnowledge 1개. 두 곳에 쓰인다.
- 2단계 합성 데이터 생성의 입력: render_as에 적힌 원천 유형(slack·ticket·guide·minutes)으로
  이 지식을 담은 원문을 만든다.
- 3단계 추출 정확도 채점 기준: 추출 결과의 entities·relations를 이 정답과 비교한다.

entities·relations·policies는 knowledge.py의 추출 모델을 그대로 쓴다.
단, 여기서 evidence 칸의 의미는 "원문 인용"이 아니라
**생성될 원문에 반드시 들어가야 할 핵심 문장**이다. 원문 생성기는 이 문장을 원문에 포함시키고,
추출기는 그 원문에서 같은 지식을 다시 뽑아내야 한다.

검증 규칙
- 관계가 가리키는 엔티티는 entities에 있어야 한다 (ExtractionResult와 같은 규칙)
- policies의 name·subject_group·system_name은 각각 entities의 Policy·UserGroup·System을
  가리켜야 한다 (정답 데이터는 스스로 일관되어야 하므로 추가로 검사)
"""

import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.schema.knowledge import (
    EntityType,
    ExtractedEntity,
    ExtractedPolicy,
    ExtractedRelation,
    NonBlankStr,
    SourceType,
    check_relation_endpoints,
)

ScenarioType = Literal["tacit_fix", "official_procedure", "policy", "policy_revision", "multihop"]


class GoldKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^K\d{3}$")
    title: NonBlankStr
    scenario_type: ScenarioType
    render_as: list[SourceType] = Field(min_length=1)
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    policies: list[ExtractedPolicy]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if len(set(self.render_as)) != len(self.render_as):
            raise ValueError("render_as에 중복 값이 있습니다")
        check_relation_endpoints(self.entities, self.relations)

        known = {(e.type, e.canonical_name) for e in self.entities}
        for i, policy in enumerate(self.policies):
            for field, entity_type, name in (
                ("name", EntityType.POLICY, policy.name),
                ("subject_group", EntityType.USER_GROUP, policy.subject_group),
                ("system_name", EntityType.SYSTEM, policy.system_name),
            ):
                if (entity_type, name) not in known:
                    raise ValueError(
                        f"policies[{i}].{field} ({entity_type}, {name!r})가 entities에 없습니다"
                    )
        return self


class _GoldYamlLoader(yaml.SafeLoader):
    """true/false만 불리언으로 읽는 SafeLoader.

    YAML 1.1은 on/off/yes/no도 불리언으로 읽어서, 관계 이름 ON이 True로 바뀐다.
    """


_GoldYamlLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_GoldYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_gold_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_GoldYamlLoader)  # SafeLoader 하위 클래스라 안전


def load_gold_knowledge(directory: str | Path) -> list[GoldKnowledge]:
    """디렉터리의 *.yaml을 파일명 순으로 읽어 검증한다. 하나라도 틀리면 파일명과 함께 실패."""
    items = []
    for path in sorted(Path(directory).glob("*.yaml")):
        data = load_gold_yaml(path.read_text(encoding="utf-8"))
        try:
            items.append(GoldKnowledge.model_validate(data))
        except ValueError as exc:  # ValidationError 포함
            raise ValueError(f"{path.name}: {exc}") from exc
    return items
