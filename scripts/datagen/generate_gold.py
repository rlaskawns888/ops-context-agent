"""정답 지식(gold) 확장 생성기.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/datagen/generate_gold.py --type tacit_fix --count 2 --dry-run
    uv run python scripts/datagen/generate_gold.py               # 유형 5종 × 11개

흐름 (유형별)
1. world.yaml·기존 gold(few-shot 1건 + 전체 목록)로 프롬프트를 만든다.
2. 최대 BATCH_SIZE개씩 Azure OpenAI에 요청한다 (strict JSON schema 구조화 출력).
   응답이 JSON·Pydantic 검증에 실패하면 오류 메시지를 붙여 최대 MAX_RETRIES회 재요청.
3. 항목마다 GoldKnowledge 검증 + gold_rules.validate_gold 검증.
   통과하면 data/gold/knowledge/Kxxx.yaml로 저장(번호는 저장 순서대로 연속),
   실패하면 사유와 함께 data/gold/_rejected/에 기록하고 모자란 개수만큼 다시 요청한다.
4. 호출마다 토큰·비용을 출력하고 data/generated/gold_usage.jsonl에 남긴다 (gold_report.py가 합산).

--dry-run: 유형별로 필요한 배치만 1회씩 호출하고 결과를 출력만 한다 (gold·_rejected 저장 안 함,
토큰 사용 기록은 dry_run 표시와 함께 남긴다).

비용 단가(USD / 1M 토큰)는 .env의 LLM_PRICE_INPUT_USD_PER_1M,
LLM_PRICE_CACHED_INPUT_USD_PER_1M, LLM_PRICE_OUTPUT_USD_PER_1M에서 읽는다.
없으면 토큰만 기록하고 비용은 "단가 미설정"으로 표시한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.schema.gold import GoldKnowledge, ScenarioType, load_gold_knowledge
from common.schema.knowledge import (
    ALLOWED_RELATIONS,
    ExtractedEntity,
    ExtractedPolicy,
    ExtractedRelation,
    NonBlankStr,
    SourceType,
    _StrictJsonSchema,
)
from common.schema.world import World, load_world
from gold_rules import core_key, system_mentions, validate_gold

ROOT = Path(__file__).resolve().parents[2]
WORLD_FILE = ROOT / "data" / "world" / "world.yaml"
GOLD_DIR = ROOT / "data" / "gold" / "knowledge"
REJECTED_DIR = ROOT / "data" / "gold" / "_rejected"
USAGE_LOG = ROOT / "data" / "generated" / "gold_usage.jsonl"

SCENARIO_TYPES: tuple[str, ...] = get_args(ScenarioType)
DEFAULT_COUNT = 11
BATCH_SIZE = 4
MAX_RETRIES = 2
MAX_COMPLETION_TOKENS = 16000
REASONING_EFFORT = "low"


# ---------------------------------------------------------------------------
# 구조화 출력 모델
# ---------------------------------------------------------------------------


class GoldDraft(BaseModel):
    """LLM이 채우는 gold 1건. id는 저장 시 부여하므로 없다.

    GoldKnowledge 그대로는 id pattern·render_as minItems 같은 키워드가 있어
    strict 호환이 보장되지 않는다. 나머지 필드는 GoldKnowledge와 같고,
    저장 전에 GoldKnowledge로 다시 검증한다.
    """

    model_config = ConfigDict(extra="forbid")

    title: NonBlankStr
    scenario_type: ScenarioType
    render_as: list[SourceType]
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    policies: list[ExtractedPolicy]


class GoldBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GoldDraft] = Field(description="요청한 개수만큼의 gold 지식")


# strict 모드가 지원하지 않거나 지원이 불확실한 키워드. 날짜 형식은 프롬프트·Pydantic으로 강제한다.
_UNSUPPORTED_KEYWORDS = frozenset({"format", "default", "minLength", "maxLength", "pattern"})


def response_json_schema() -> dict[str, Any]:
    """GoldBatch의 strict JSON 스키마 (Azure OpenAI response_format용)."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in _UNSUPPORTED_KEYWORDS or k == "properties"
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(GoldBatch.model_json_schema(schema_generator=_StrictJsonSchema))


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

_COMMON_RULES = f"""\
너는 사내 IT 운영 지식그래프의 정답 데이터(gold)를 만드는 데이터 설계자다.
gold 1건은 하나의 지식이며, 이후 이 gold를 근거로 원문(slack·ticket·guide·minutes)이 생성되고
추출기가 그 원문에서 같은 엔티티·관계를 다시 뽑아낸다. 정확성과 일관성이 가장 중요하다.

[이름 규칙]
- System·Team·UserGroup·Permission의 canonical_name은 world의 canonical_name을 글자 그대로 쓴다.
- 이들의 aliases는 world에 적힌 그 항목의 aliases 중에서만 고른다 (없으면 빈 배열). 새 별칭 금지.
- Permission.system_name은 world의 on_system과 같아야 하고, Permission ON System 관계를 넣는다.
- Error.code는 world error_codes의 code 중 하나이거나 null(코드 없는 증상)이다.
  code가 있으면 canonical_name도 code와 같게 쓰고, OCCURS_IN 대상은 world에서 그 오류의 system이다.
  symptom과 해결법은 그 code의 world message·description이 말하는 원인과 맞아야 한다
  (예: E-901은 쓰기 권한 없음이므로 권한 신청이 해결법이고, 토큰 재발급은 E-902의 해결법이다).
- DEPENDS_ON·OWNED_BY 관계는 world의 depends_on·owner_team과 정확히 일치해야 한다.
- Solution·Policy·코드 없는 Error의 이름은 자유롭게 짓되 짧고 구체적인 한국어로 쓴다.

[관계 규칙] (source_type, relation_type, target_type) 허용 조합은 아래뿐이다.
{chr(10).join(f"- {s} -{r}-> {t}" for s, r, t in sorted(ALLOWED_RELATIONS))}
- 관계의 source·target은 반드시 같은 gold의 entities에 (type, canonical_name)으로 있어야 한다.

[정책 규칙]
- Policy 엔티티마다 같은 이름의 policies 항목을 1개 둔다. 둘의 valid_from·valid_to는 같아야 한다.
- policies의 subject_group·system_name은 entities의 UserGroup·System canonical_name이고,
  (system_name, level)이 같은 Permission 엔티티가 entities에 있어야 한다.

[날짜] YYYY-MM-DD 형식, 2025-01-01 ~ 2026-11-30 범위만. 기간이 없으면 null.

[evidence] 생성될 원문에 그대로 들어갈 핵심 문장이다. 한국어 자연문 1문장, 비워 두지 않는다.
관계의 evidence는 그 관계를 직접 말하는 문장이어야 한다.

[다양성]
- 이미 있는 gold와 (유형, 핵심 엔티티 조합)이 겹치면 안 된다.
  핵심 엔티티 = System·Error·Permission·UserGroup·Team.
- 한 번에 여러 건을 만들 때 서로도 겹치지 않게 한다.
- world 시스템 중 등장 횟수가 적은 시스템을 우선 사용한다.
"""

_TYPE_RULES: dict[str, str] = {
    "tacit_fix": """\
[유형: tacit_fix — 암묵지 해결법]
공식 문서에는 없고 현업 대화·티켓에서만 전해지는 우회/해결법.
- Error OCCURS_IN System, Error RESOLVED_BY Solution 관계를 반드시 넣는다.
- Solution.knowledge_type = tacit. steps는 2~4단계. valid_from·valid_to는 보통 null.
- render_as는 slack·ticket 중에서 고른다 (guide는 공식 문서라 쓰지 않는다).""",
    "official_procedure": """\
[유형: official_procedure — 공식 처리 절차]
가이드 문서에 공식으로 정리된 처리 절차.
- Error OCCURS_IN System, Error RESOLVED_BY Solution 관계를 반드시 넣는다.
- Solution.knowledge_type = official. steps는 3~5단계.
- render_as에 guide를 반드시 포함한다 (추가로 slack·ticket 가능).""",
    "policy": """\
[유형: policy — 권한 정책]
특정 사용자 그룹이 특정 권한을 가질 수 있는지 정하는 정책 1개.
- UserGroup, Permission, 그 Permission의 System, Policy 엔티티를 넣는다.
- Permission ON System, Policy GOVERNS Permission, Policy APPLIES_TO UserGroup 관계를 넣는다.
- policies에 효과(ALLOW·DENY·REQUIRE_APPROVAL)를 적는다. REQUIRE_APPROVAL이면 conditions에
  누구의 승인인지 쓴다. SUPERSEDES는 쓰지 않는다 (policy_revision 유형 전용).
- 효과 3종이 골고루 나오게 한다. render_as는 guide·ticket·minutes·slack 중 1~2개.""",
    "policy_revision": """\
[유형: policy_revision — 정책 개정]
같은 (UserGroup, Permission)에 대한 이전 정책이 신규 정책으로 바뀐 이력.
- Policy 엔티티 2개(이전·신규), 각각 GOVERNS·APPLIES_TO 관계, 신규 SUPERSEDES 이전 관계를 넣는다.
- 두 Policy 모두 canonical_name 끝에 " (시행일)"을 붙인다. 예: "외주 운영DB 쓰기 금지 (2026-01-01)".
  시행일은 그 정책의 valid_from과 같다.
- 이전 정책 valid_to = 신규 정책 valid_from의 전날 (예: 신규 2026-03-01 → 이전 2026-02-28).
  신규 정책 valid_to는 보통 null.
- policies에 두 정책을 모두 적는다. 효과나 조건이 실제로 바뀌어야 한다.
- render_as에 guide 또는 minutes를 포함한다.""",
    "multihop": """\
[유형: multihop — 여러 단계를 거쳐야 답이 나오는 지식]
관계를 2번 이상 따라가야 답할 수 있는 지식 (예: 오류 → 발생 시스템 → 의존 시스템 → 담당 팀).
- 관계 방향(source→target)을 따라 2홉 이상 이어지는 경로가 반드시 있어야 한다.
  예: Error OCCURS_IN System, System DEPENDS_ON System, System OWNED_BY Team.
- DEPENDS_ON·OWNED_BY는 world 정의 그대로만 쓴다. Team.channel은 모르면 null.
- render_as는 slack·minutes·ticket 중 1~2개.""",
}


def _gold_line(item: GoldKnowledge) -> str:
    names = sorted(n for _, n in core_key(item)[1])
    return f"- {item.id} [{item.scenario_type}] {item.title} | 핵심: {', '.join(names)}"


def world_prompt(world: World) -> str:
    return yaml.safe_dump(
        world.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=200
    )


def build_messages(
    scenario: str,
    count: int,
    world: World,
    existing: Sequence[GoldKnowledge],
    recent_rejections: Sequence[str] = (),
) -> list[dict[str, str]]:
    examples = [g for g in existing if g.scenario_type == scenario][:1]
    coverage = system_mentions(existing)
    coverage_lines = ", ".join(
        f"{s.canonical_name}={coverage.get(s.canonical_name, 0)}" for s in world.systems
    )
    system = "\n\n".join([_COMMON_RULES, _TYPE_RULES[scenario], "[world]\n" + world_prompt(world)])
    parts = [
        f"scenario_type이 {scenario}인 gold 지식을 정확히 {count}개 만들어 items에 담아라.",
        "[예시 gold (형식·수준 참고, 내용 복제 금지)]\n"
        + "\n---\n".join(dump_gold_yaml(g) for g in examples),
        "[이미 있는 gold 목록 — 이것들과 핵심 엔티티 조합이 겹치지 않게]\n"
        + "\n".join(_gold_line(g) for g in existing),
        f"[world 시스템별 현재 등장 횟수]\n{coverage_lines}",
    ]
    if recent_rejections:
        parts.append(
            "[직전 생성에서 검증에 실패한 사유 — 같은 실수를 반복하지 말 것]\n"
            + "\n".join(f"- {r}" for r in recent_rejections)
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


# ---------------------------------------------------------------------------
# YAML 저장
# ---------------------------------------------------------------------------

_DATE_FIELDS = {"valid_from", "valid_to"}


def _restore_dates(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: date.fromisoformat(v)
            if k in _DATE_FIELDS and isinstance(v, str)
            else _restore_dates(v)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_restore_dates(v) for v in node]
    return node


def dump_gold_yaml(item: GoldKnowledge) -> str:
    """기존 K001~K005와 같은 모양의 YAML (엔티티는 type이 맨 앞, 날짜는 따옴표 없이)."""
    data = _restore_dates(item.model_dump(mode="json"))
    data["entities"] = [{"type": e["type"], **e} for e in data["entities"]]
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200)


def next_gold_number(gold_dir: Path) -> int:
    numbers = [int(p.stem[1:]) for p in gold_dir.glob("K[0-9][0-9][0-9].yaml")]
    return max(numbers, default=0) + 1


# ---------------------------------------------------------------------------
# 비용
# ---------------------------------------------------------------------------


class PriceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_price_input_usd_per_1m: float | None = None
    llm_price_cached_input_usd_per_1m: float | None = None
    llm_price_output_usd_per_1m: float | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        u = getattr(response, "usage", None)
        if u is None:
            return cls()
        prompt_details = getattr(u, "prompt_tokens_details", None)
        completion_details = getattr(u, "completion_tokens_details", None)
        return cls(
            prompt_tokens=u.prompt_tokens or 0,
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            completion_tokens=u.completion_tokens or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )

    def cost_usd(self, prices: PriceSettings) -> float | None:
        p_in = prices.llm_price_input_usd_per_1m
        p_out = prices.llm_price_output_usd_per_1m
        if p_in is None or p_out is None:
            return None
        p_cached = prices.llm_price_cached_input_usd_per_1m
        p_cached = p_in if p_cached is None else p_cached
        uncached = self.prompt_tokens - self.cached_tokens
        return (
            uncached * p_in + self.cached_tokens * p_cached + self.completion_tokens * p_out
        ) / 1e6


def format_cost(cost: float | None) -> str:
    return "단가 미설정" if cost is None else f"${cost:.4f}"


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------


class GenerationError(Exception):
    pass


@dataclass
class Rejection:
    draft: dict[str, Any]
    reasons: list[str]


@dataclass
class RunState:
    world: World
    gold: list[GoldKnowledge]  # 기존 + 이번에 통과한 것 (중복 판정·프롬프트 목록용)
    next_number: int
    accepted: list[GoldKnowledge] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


def request_batch(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    *,
    scenario: str,
    log_call: Any,
) -> GoldBatch:
    """1회 요청 + JSON·Pydantic 실패 시 오류를 붙여 최대 MAX_RETRIES회 재요청."""
    messages = list(messages)
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            name=f"generate_gold.{scenario}",
            model=deployment,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "gold_batch",
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        log_call(Usage.from_response(response), attempt)
        choice = response.choices[0]
        content = choice.message.content or ""
        if getattr(choice.message, "refusal", None):
            last_error = f"모델 거부: {choice.message.refusal}"
        elif choice.finish_reason == "length":
            last_error = "출력이 max_completion_tokens에서 잘렸습니다. 더 간결하게 출력하세요."
        else:
            try:
                return GoldBatch.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)
        print(f"  [retry {attempt + 1}/{MAX_RETRIES}] 응답 검증 실패: {last_error[:300]}")
        messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": f"검증 실패. 고쳐서 전체 JSON을 다시 출력하라.\n{last_error}",
            },
        ]
    raise GenerationError(f"재시도 {MAX_RETRIES}회 후에도 검증 실패: {last_error[:500]}")


def check_draft(
    draft: GoldDraft, scenario: str, state: RunState
) -> tuple[GoldKnowledge | None, list[str]]:
    """draft → GoldKnowledge 변환 + 규칙 검증. 통과하면 (gold, []), 실패하면 (None, 사유)."""
    if draft.scenario_type != scenario:
        return None, [f"요청 유형 {scenario}와 다른 scenario_type {draft.scenario_type}"]
    gold_id = f"K{state.next_number:03d}"
    try:
        item = GoldKnowledge.model_validate({"id": gold_id, **draft.model_dump()})
    except ValidationError as exc:
        return None, [
            f"스키마: {err['msg']} @ {'.'.join(map(str, err['loc']))}" for err in exc.errors()
        ]
    errors = validate_gold(item, state.world, state.gold)
    return (None, errors) if errors else (item, [])


def generate_type(
    client: Any,
    deployment: str,
    scenario: str,
    count: int,
    state: RunState,
    *,
    dry_run: bool,
    log_call: Any,
) -> None:
    """scenario 유형을 count개 통과시킬 때까지 생성한다 (dry_run이면 배치 1바퀴만)."""
    start = len(state.accepted)
    target = start + count
    max_calls = -(-count // BATCH_SIZE) * 3
    calls = 0
    recent: list[str] = []
    while len(state.accepted) < target and calls < max_calls:
        need = min(BATCH_SIZE, target - len(state.accepted))
        if dry_run and calls * BATCH_SIZE >= count:
            break
        calls += 1
        messages = build_messages(scenario, need, state.world, state.gold, recent)
        try:
            batch = request_batch(
                client, deployment, messages, scenario=scenario, log_call=log_call
            )
        except GenerationError as exc:
            print(f"  [FAIL] {exc}")
            continue
        recent = []
        for draft in batch.items[:need]:
            item, reasons = check_draft(draft, scenario, state)
            if item is None:
                state.rejected.append(Rejection(draft.model_dump(mode="json"), reasons))
                recent.extend(reasons)
                print(f"  [REJECT] {draft.title}")
                for r in reasons:
                    print(f"      - {r}")
                continue
            state.accepted.append(item)
            state.gold.append(item)
            state.next_number += 1
            print(f"  [OK] {item.id} {item.title}")
    if len(state.accepted) < target:
        print(f"  [WARN] {scenario}: 목표 {count}개 중 {len(state.accepted) - start}개만 통과")


# ---------------------------------------------------------------------------
# 저장·로그
# ---------------------------------------------------------------------------


def save_accepted(items: Sequence[GoldKnowledge], gold_dir: Path) -> None:
    gold_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        path = gold_dir / f"{item.id}.yaml"
        if path.exists():
            raise FileExistsError(f"{path} 이미 존재 (기존 gold는 덮어쓰지 않음)")
        path.write_text(dump_gold_yaml(item), encoding="utf-8")


def save_rejected(
    rejections: Sequence[Rejection], rejected_dir: Path, run_id: str, offset: int = 0
) -> None:
    if not rejections:
        return
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for i, rej in enumerate(rejections, offset + 1):
        scenario = rej.draft.get("scenario_type", "unknown")
        path = rejected_dir / f"{run_id}-{i:02d}-{scenario}.yaml"
        data = {"reasons": rej.reasons, "draft": _restore_dates(rej.draft)}
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8"
        )


def append_usage(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_client() -> tuple[Any, Any, str]:
    from common.settings import get_settings

    settings = get_settings()
    required = [
        "azure_openai_endpoint",
        "azure_openai_api_key",
        "azure_openai_api_version",
        "azure_openai_chat_deployment",
        "langfuse_host",
        "langfuse_public_key",
        "langfuse_secret_key",
    ]
    missing = [f.upper() for f in required if not getattr(settings, f)]
    if missing:
        raise SystemExit(f".env 미설정: {', '.join(missing)}")

    from langfuse import Langfuse
    from langfuse.openai import AzureOpenAI  # Langfuse 래퍼: 모든 호출이 자동 추적된다

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
    )
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
    )
    return client, langfuse, settings.azure_openai_chat_deployment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gold 지식 확장 생성")
    parser.add_argument("--type", choices=SCENARIO_TYPES, help="생성할 유형 (생략 시 5종 전부)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="유형별 추가 개수")
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count는 1 이상")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = [args.type] if args.type else list(SCENARIO_TYPES)
    world = load_world(WORLD_FILE)
    gold = load_gold_knowledge(GOLD_DIR)
    state = RunState(world=world, gold=list(gold), next_number=next_gold_number(GOLD_DIR))
    prices = PriceSettings()
    client, langfuse, deployment = _make_client()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    totals = Usage()
    total_cost: list[float | None] = []

    def make_logger(scenario: str):
        def log_call(usage: Usage, attempt: int) -> None:
            cost = usage.cost_usd(prices)
            total_cost.append(cost)
            for f in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
                setattr(totals, f, getattr(totals, f) + getattr(usage, f))
            print(
                f"  [call] {scenario} attempt={attempt} in={usage.prompt_tokens} "
                f"(cached {usage.cached_tokens}) out={usage.completion_tokens} "
                f"(reasoning {usage.reasoning_tokens}) cost={format_cost(cost)}"
            )
            append_usage(
                USAGE_LOG,
                {
                    "run_id": run_id,
                    "ts": datetime.now(UTC).isoformat(),
                    "scenario_type": scenario,
                    "model": deployment,
                    "attempt": attempt,
                    "dry_run": args.dry_run,
                    **usage.__dict__,
                    "cost_usd": cost,
                },
            )

        return log_call

    started = time.monotonic()
    saved = rejected_saved = 0
    try:
        for scenario in scenarios:
            print(f"== {scenario} × {args.count}{' (dry-run)' if args.dry_run else ''}")
            generate_type(
                client,
                deployment,
                scenario,
                args.count,
                state,
                dry_run=args.dry_run,
                log_call=make_logger(scenario),
            )
            if not args.dry_run:  # 유형마다 저장해 중간에 중단돼도 결과가 남는다
                save_accepted(state.accepted[saved:], GOLD_DIR)
                save_rejected(state.rejected[rejected_saved:], REJECTED_DIR, run_id, rejected_saved)
                saved, rejected_saved = len(state.accepted), len(state.rejected)
    finally:
        langfuse.flush()

    if args.dry_run:
        for item in state.accepted:
            print(f"\n# --- {item.id} (dry-run, 저장 안 함) ---\n{dump_gold_yaml(item)}")

    cost = (
        None if any(c is None for c in total_cost) else sum(c for c in total_cost if c is not None)
    )
    print(
        f"\n통과 {len(state.accepted)} / 거부 {len(state.rejected)} | "
        f"토큰 in={totals.prompt_tokens} (cached {totals.cached_tokens}) "
        f"out={totals.completion_tokens} (reasoning {totals.reasoning_tokens}) | "
        f"비용 {format_cost(cost)} | {time.monotonic() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
