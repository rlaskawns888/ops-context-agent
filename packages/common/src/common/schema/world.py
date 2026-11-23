"""합성 데이터 세계관(world).

모든 합성 문서(gold, slack, ticket, guide, minutes)가 공유하는 고정 명칭 사전.
생성 프롬프트는 항상 이 세계관을 참조해서 시스템·팀·그룹·권한·오류코드 이름을 고른다.
원본은 data/world/world.yaml 하나뿐이다.

검증 규칙 (load 시 실행)
- 이름 참조(owner_team, depends_on, on_system, error.system)는 정의된 canonical_name이어야 한다
- canonical_name·aliases·오류 code는 세계관 전체에서 중복될 수 없다
  (대소문자·공백 차이는 같은 이름으로 본다: "운영 DB" == "운영db")
- depends_on에 순환이 없어야 한다 (자기 자신 의존 포함)
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from common.schema.gold import load_gold_yaml
from common.schema.knowledge import NonBlankStr


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class _WorldModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SystemDef(_WorldModel):
    canonical_name: NonBlankStr
    aliases: list[NonBlankStr]
    owner_team: NonBlankStr
    depends_on: list[NonBlankStr]
    description: NonBlankStr


class TeamDef(_WorldModel):
    canonical_name: NonBlankStr
    aliases: list[NonBlankStr]
    description: NonBlankStr


class UserGroupDef(_WorldModel):
    canonical_name: NonBlankStr
    aliases: list[NonBlankStr]
    description: NonBlankStr


class PermissionDef(_WorldModel):
    canonical_name: NonBlankStr
    aliases: list[NonBlankStr]
    on_system: NonBlankStr
    risk_level: RiskLevel


class ErrorCodeDef(_WorldModel):
    code: NonBlankStr
    system: NonBlankStr
    message: NonBlankStr
    description: NonBlankStr


NameKind = Literal["system", "team", "user_group", "permission", "error_code"]


@dataclass(frozen=True)
class NameRef:
    """세계관 안의 이름 하나가 가리키는 대상."""

    kind: NameKind
    canonical_name: str  # error_code는 code
    is_alias: bool


def normalize_name(name: str) -> str:
    """중복 판정용 키. 대소문자와 모든 공백을 무시한다."""
    return "".join(name.split()).casefold()


class World(_WorldModel):
    systems: list[SystemDef]
    teams: list[TeamDef]
    user_groups: list[UserGroupDef]
    permissions: list[PermissionDef]
    error_codes: list[ErrorCodeDef]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        self._check_unique_names()
        self._check_references()
        self._check_no_dependency_cycle()
        return self

    # -- 이름 색인 ----------------------------------------------------------

    def _iter_names(self) -> list[tuple[str, NameRef]]:
        named: list[tuple[NameKind, list[SystemDef | TeamDef | UserGroupDef | PermissionDef]]] = [
            ("system", self.systems),
            ("team", self.teams),
            ("user_group", self.user_groups),
            ("permission", self.permissions),
        ]
        names: list[tuple[str, NameRef]] = []
        for kind, items in named:
            for item in items:
                names.append((item.canonical_name, NameRef(kind, item.canonical_name, False)))
                names.extend(
                    (alias, NameRef(kind, item.canonical_name, True)) for alias in item.aliases
                )
        names.extend((e.code, NameRef("error_code", e.code, False)) for e in self.error_codes)
        return names

    def lookup(self, name: str) -> NameRef | None:
        """이름(대표 이름·별칭·오류 코드)이 가리키는 대상. 없으면 None."""
        key = normalize_name(name)
        for candidate, ref in self._iter_names():
            if normalize_name(candidate) == key:
                return ref
        return None

    # -- 검증 ---------------------------------------------------------------

    def _check_unique_names(self) -> None:
        seen: dict[str, tuple[str, NameRef]] = {}
        for name, ref in self._iter_names():
            key = normalize_name(name)
            if key in seen:
                prev_name, prev_ref = seen[key]
                raise ValueError(
                    f"이름 중복: {name!r} ({ref.kind} {ref.canonical_name!r})가 "
                    f"{prev_name!r} ({prev_ref.kind} {prev_ref.canonical_name!r})와 겹칩니다"
                )
            seen[key] = (name, ref)

    def _check_references(self) -> None:
        systems = {s.canonical_name for s in self.systems}
        teams = {t.canonical_name for t in self.teams}

        def require(known: set[str], name: str, where: str, kind: str) -> None:
            if name not in known:
                raise ValueError(f"{where}: {name!r}는 정의된 {kind} canonical_name이 아닙니다")

        for s in self.systems:
            require(teams, s.owner_team, f"system {s.canonical_name!r}.owner_team", "team")
            if len(set(s.depends_on)) != len(s.depends_on):
                raise ValueError(f"system {s.canonical_name!r}.depends_on에 중복 값이 있습니다")
            for dep in s.depends_on:
                require(systems, dep, f"system {s.canonical_name!r}.depends_on", "system")
        for p in self.permissions:
            require(systems, p.on_system, f"permission {p.canonical_name!r}.on_system", "system")
        for e in self.error_codes:
            require(systems, e.system, f"error_code {e.code!r}.system", "system")

    def _check_no_dependency_cycle(self) -> None:
        graph = {s.canonical_name: s.depends_on for s in self.systems}
        done: set[str] = set()

        def visit(node: str, path: list[str]) -> None:
            if node in path:
                cycle = [*path[path.index(node) :], node]
                raise ValueError(f"depends_on 순환: {' -> '.join(cycle)}")
            if node in done:
                return
            for dep in graph[node]:
                visit(dep, [*path, node])
            done.add(node)

        for name in graph:
            visit(name, [])

    # -- 조회 ---------------------------------------------------------------

    def dependency_chain(self, system: str) -> list[list[str]]:
        """system에서 시작하는 모든 depends_on 경로 (말단까지). 의존이 없으면 [[system]]."""
        graph = {s.canonical_name: s.depends_on for s in self.systems}
        deps = graph[system]
        if not deps:
            return [[system]]
        return [[system, *rest] for dep in deps for rest in self.dependency_chain(dep)]


def load_world(path: str | Path) -> World:
    """world.yaml을 읽어 검증한다. 틀리면 파일명과 함께 ValueError."""
    path = Path(path)
    data = load_gold_yaml(path.read_text(encoding="utf-8"))
    try:
        return World.model_validate(data)
    except ValueError as exc:  # ValidationError 포함
        raise ValueError(f"{path.name}: {exc}") from exc
