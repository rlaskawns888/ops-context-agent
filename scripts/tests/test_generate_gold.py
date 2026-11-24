"""gold 확장 생성: 검증 규칙·생성 루프·리포트 (LLM 호출 없음)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import generate_gold
import gold_report
import gold_rules
from common.schema.gold import GoldKnowledge, load_gold_yaml
from common.schema.world import load_world

WORLD = load_world(generate_gold.WORLD_FILE)


def _seed(gold_id: str) -> dict[str, Any]:
    """기존 gold 파일을 dict로 (테스트마다 복사본을 고쳐 쓴다)."""
    path = generate_gold.GOLD_DIR / f"{gold_id}.yaml"
    return load_gold_yaml(path.read_text(encoding="utf-8"))


def _gold(data: dict[str, Any], gold_id: str = "K900") -> GoldKnowledge:
    return GoldKnowledge.model_validate({**data, "id": gold_id})


def _errors(data: dict[str, Any], existing: list[GoldKnowledge] = ()) -> list[str]:
    return gold_rules.validate_gold(_gold(data), WORLD, existing)


def _entity(data: dict[str, Any], entity_type: str, name: str | None = None) -> dict[str, Any]:
    return next(
        e
        for e in data["entities"]
        if e["type"] == entity_type and (name is None or e["canonical_name"] == name)
    )


def _rename(data: dict[str, Any], old: str, new: str) -> None:
    for e in data["entities"]:
        if e["canonical_name"] == old:
            e["canonical_name"] = new
    for r in data["relations"]:
        for key in ("source_name", "target_name"):
            if r[key] == old:
                r[key] = new
    for p in data["policies"]:
        if p["name"] == old:
            p["name"] = new


def _has(errors: list[str], fragment: str) -> bool:
    return any(fragment in e for e in errors)


# ---------------------------------------------------------------------------
# 유형별 규칙: 통과 1건 + 실패 1건
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_id", ["K001", "K002", "K003", "K004", "K005"])
def test_seed_gold_passes_all_rules(gold_id: str) -> None:
    assert _errors(_seed(gold_id)) == []


def test_tacit_fix_requires_tacit_solution() -> None:
    data = _seed("K001")
    _entity(data, "Solution")["knowledge_type"] = "official"
    assert _has(_errors(data), "tacit_fix: Solution 'PC 재부팅'.knowledge_type이 tacit이 아님")


def test_official_procedure_requires_guide_render() -> None:
    data = _seed("K002")
    data["render_as"] = ["slack", "ticket"]
    assert _has(_errors(data), "render_as에 guide가 없음")


def test_official_procedure_rejects_tacit_solution() -> None:
    data = _seed("K002")
    _entity(data, "Solution")["knowledge_type"] = "tacit"
    assert _has(_errors(data), "official_procedure: Solution '잠금 해제 절차'가 tacit")


def test_policy_requires_policies_with_effect() -> None:
    data = _seed("K003")
    data["policies"] = []
    errors = _errors(data)
    assert _has(errors, "policy: policies(효과")
    assert _has(errors, "대응하는 policies 항목이 없음")


def test_policy_rejects_supersedes() -> None:
    data = _seed("K004")
    data["scenario_type"] = "policy"
    assert _has(_errors(data), "SUPERSEDES는 policy_revision 유형에서만")


def test_policy_revision_requires_old_valid_to_day_before_new_valid_from() -> None:
    data = _seed("K004")
    old = "해외 VPN 접속 허용 (2025-01-01)"
    _entity(data, "Policy", old)["valid_to"] = "2026-01-01"
    next(p for p in data["policies"] if p["name"] == old)["valid_to"] = "2026-01-01"
    assert _has(_errors(data), "신규 정책 시행일 전날 2025-12-31")


def test_policy_revision_requires_effective_date_in_name() -> None:
    data = _seed("K004")
    _rename(data, "해외 VPN 접속 승인 필요 (2026-01-01)", "해외 VPN 접속 승인 필요")
    assert _has(_errors(data), "이름 끝에 시행일 (2026-01-01)이 없음")


def test_policy_revision_requires_supersedes() -> None:
    data = _seed("K004")
    data["relations"] = [r for r in data["relations"] if r["relation_type"] != "SUPERSEDES"]
    assert _has(_errors(data), "SUPERSEDES 관계가 없음")


def test_multihop_requires_two_hop_path() -> None:
    data = _seed("K005")
    data["relations"] = [r for r in data["relations"] if r["relation_type"] != "OWNED_BY"]
    assert gold_rules.longest_path(_gold(data)) == 1
    assert _has(_errors(data), "multihop: 관계 방향을 따라 2홉 이상")


# ---------------------------------------------------------------------------
# world 대조·날짜·중복
# ---------------------------------------------------------------------------


def test_name_not_in_world_is_rejected() -> None:
    data = _seed("K001")
    _rename(data, "VPN", "사내 VPN")  # world에서는 별칭
    errors = _errors(data)
    assert _has(errors, "System '사내 VPN': world canonical_name이 아님 (world에서는 system 'VPN')")


def test_alias_outside_world_aliases_is_rejected() -> None:
    data = _seed("K001")
    _entity(data, "System")["aliases"] = ["사내 VPN", "VPN망"]
    assert _has(_errors(data), "world aliases에 없는 별칭 ['VPN망']")


def test_unknown_error_code_is_rejected() -> None:
    data = _seed("K001")
    _entity(data, "Error")["code"] = "E-199"
    assert _has(_errors(data), "code 'E-199'가 world에 없음")


def test_error_code_must_occur_in_its_world_system() -> None:
    data = _seed("K001")
    _entity(data, "Error")["code"] = "E-201"  # world에서는 MFA 오류
    assert _has(_errors(data), "Error E-201: OCCURS_IN 대상 ['VPN'] ≠ world system ['MFA']")


def test_relations_must_match_world_dependencies_and_owner() -> None:
    data = _seed("K005")
    owned = next(r for r in data["relations"] if r["relation_type"] == "OWNED_BY")
    owned["source_name"] = "VPN"  # world: VPN 담당은 네트워크팀
    errors = _errors(data)
    assert _has(errors, "VPN OWNED_BY 보안운영팀: world owner_team은 '네트워크팀'")


def test_date_out_of_range_is_rejected() -> None:
    data = _seed("K004")
    old = "해외 VPN 접속 허용 (2025-01-01)"
    _entity(data, "Policy", old)["valid_from"] = "2024-12-31"
    next(p for p in data["policies"] if p["name"] == old)["valid_from"] = "2024-12-31"
    assert _has(_errors(data), "valid_from 2024-12-31가 2025-01-01~2026-11-30 범위 밖")


def test_duplicate_core_entities_are_rejected() -> None:
    existing = _gold(_seed("K001"), "K001")
    data = _seed("K001")
    data["title"] = "다른 제목이어도 같은 오류·시스템 조합이면 중복"
    _rename(data, "PC 재부팅", "VPN 클라이언트 재설치")  # Solution 이름은 핵심 엔티티가 아님
    assert _has(_errors(data, [existing]), "중복: K001")


def test_same_core_entities_in_different_type_is_not_duplicate() -> None:
    existing = _gold(_seed("K001"), "K001")
    data = _seed("K001")
    data["scenario_type"] = "official_procedure"
    data["render_as"] = ["guide"]
    _entity(data, "Solution")["knowledge_type"] = "official"
    assert _errors(data, [existing]) == []


# ---------------------------------------------------------------------------
# 생성기: 스키마·YAML·재시도·생성 루프
# ---------------------------------------------------------------------------


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_response_schema_is_strict_compatible() -> None:
    schema = generate_gold.response_json_schema()
    text = json.dumps(schema)
    for keyword in ('"oneOf"', '"const"', '"discriminator"', '"format"', '"pattern"'):
        assert keyword not in text
    for node in _walk(schema):
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


@pytest.mark.parametrize("gold_id", ["K001", "K003", "K004", "K005"])
def test_dump_gold_yaml_round_trips(gold_id: str) -> None:
    item = _gold(_seed(gold_id), gold_id)
    text = generate_gold.dump_gold_yaml(item)
    assert text.startswith(f"id: {gold_id}\n")
    assert "- type: " in text  # 엔티티는 type이 맨 앞
    assert GoldKnowledge.model_validate(load_gold_yaml(text)) == item


def test_next_gold_number(tmp_path: Path) -> None:
    assert generate_gold.next_gold_number(tmp_path) == 1
    (tmp_path / "K007.yaml").write_text("", encoding="utf-8")
    (tmp_path / "K003.yaml").write_text("", encoding="utf-8")
    assert generate_gold.next_gold_number(tmp_path) == 8


def _draft(gold_id: str) -> dict[str, Any]:
    data = _seed(gold_id)
    data.pop("id")
    return data


def _response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=40,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
    )


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return _response(self.contents.pop(0))


def _batch(*drafts: dict[str, Any]) -> str:
    return json.dumps({"items": list(drafts)}, ensure_ascii=False, default=str)


def test_request_batch_retries_with_error_message() -> None:
    client = FakeClient(["{not json", _batch(_draft("K001"))])
    logged: list[int] = []
    batch = generate_gold.request_batch(
        client,
        "dep",
        [{"role": "user", "content": "x"}],
        scenario="tacit_fix",
        log_call=lambda usage, attempt: logged.append(attempt),
    )
    assert len(batch.items) == 1
    assert logged == [0, 1]
    first, second = client.calls
    assert first["reasoning_effort"] == "low"
    assert first["response_format"]["json_schema"]["strict"] is True
    assert second["messages"][-1]["content"].startswith("검증 실패")


def test_request_batch_gives_up_after_max_retries() -> None:
    client = FakeClient(["{bad"] * (generate_gold.MAX_RETRIES + 1))
    with pytest.raises(generate_gold.GenerationError):
        generate_gold.request_batch(
            client, "dep", [], scenario="tacit_fix", log_call=lambda *a: None
        )
    assert len(client.calls) == generate_gold.MAX_RETRIES + 1


def _state(next_number: int = 6) -> generate_gold.RunState:
    return generate_gold.RunState(world=WORLD, gold=[], next_number=next_number)


def test_generate_type_accepts_valid_and_rejects_duplicate_and_wrong_type() -> None:
    wrong_type = _draft("K003")
    client = FakeClient(
        [_batch(_draft("K001"), _draft("K001"), wrong_type), _batch(_draft("K002"))]
    )
    state = _state()
    generate_gold.generate_type(
        client, "dep", "tacit_fix", 3, state, dry_run=True, log_call=lambda *a: None
    )
    assert [g.id for g in state.accepted] == ["K006"]
    reasons = [r for rej in state.rejected for r in rej.reasons]
    assert _has(reasons, "중복: K006")
    assert _has(reasons, "다른 scenario_type policy")
    assert len(client.calls) == 1  # dry-run은 배치 1바퀴만, 모자라도 다시 요청하지 않는다


def test_generate_type_refills_rejected_items_and_numbers_sequentially() -> None:
    bad = _draft("K001")
    bad["entities"][1]["code"] = "E-199"  # world에 없는 코드
    good = _draft("K001")
    client = FakeClient([_batch(bad), _batch(good)])
    state = _state(next_number=42)
    generate_gold.generate_type(
        client, "dep", "tacit_fix", 1, state, dry_run=False, log_call=lambda *a: None
    )
    assert [g.id for g in state.accepted] == ["K042"]
    assert len(state.rejected) == 1
    retry_prompt = client.calls[1]["messages"][-1]["content"]
    assert "E-199" in retry_prompt  # 직전 거부 사유를 다음 프롬프트에 넣는다


def test_save_accepted_and_rejected(tmp_path: Path) -> None:
    item = _gold(_seed("K001"), "K006")
    generate_gold.save_accepted([item], tmp_path / "knowledge")
    saved = load_gold_yaml((tmp_path / "knowledge" / "K006.yaml").read_text(encoding="utf-8"))
    assert GoldKnowledge.model_validate(saved) == item
    with pytest.raises(FileExistsError):
        generate_gold.save_accepted([item], tmp_path / "knowledge")

    rej = generate_gold.Rejection({"scenario_type": "policy", "title": "t"}, ["사유"])
    generate_gold.save_rejected([rej], tmp_path / "_rejected", "RUN", offset=2)
    text = (tmp_path / "_rejected" / "RUN-03-policy.yaml").read_text(encoding="utf-8")
    assert load_gold_yaml(text)["reasons"] == ["사유"]


def test_usage_cost() -> None:
    usage = generate_gold.Usage(prompt_tokens=1_000_000, cached_tokens=500_000, completion_tokens=0)
    prices = generate_gold.PriceSettings(
        _env_file=None,
        llm_price_input_usd_per_1m=2.0,
        llm_price_cached_input_usd_per_1m=0.5,
        llm_price_output_usd_per_1m=8.0,
    )
    assert usage.cost_usd(prices) == pytest.approx(1.25)
    assert usage.cost_usd(generate_gold.PriceSettings(_env_file=None)) is None


def test_build_messages_includes_existing_list_and_rejections() -> None:
    existing = [_gold(_seed("K001"), "K001"), _gold(_seed("K003"), "K003")]
    messages = generate_gold.build_messages("tacit_fix", 2, WORLD, existing, ["사유A"])
    user = messages[1]["content"]
    assert "정확히 2개" in user
    assert "K003 [policy]" in user  # 다른 유형도 목록에 들어간다
    assert "사유A" in user
    assert "canonical_name: 망연계" in messages[0]["content"]  # world 포함


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------


def test_report_counts_and_unused_systems() -> None:
    gold = [_gold(_seed(f"K00{i}"), f"K00{i}") for i in range(1, 6)]
    usage = [
        {"prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 5, "cost_usd": 0.01},
        {"prompt_tokens": 20, "cached_tokens": 5, "completion_tokens": 5, "cost_usd": None},
    ]
    report = gold_report.build_report(WORLD, gold, usage, rejected_count=3)
    assert "gold 총 5건 (K001~K005)" in report
    assert "tacit_fix            1" in report
    assert "VPN                  3" in report
    assert "미등장 시스템: SSO, 메일" in report
    assert "_rejected/ 3건" in report
    assert "호출 2회" in report
    assert "단가 미설정 호출 1회" in report
