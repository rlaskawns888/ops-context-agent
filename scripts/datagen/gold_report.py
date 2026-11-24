"""gold 지식 요약 리포트.

실행:
    uv run python scripts/datagen/gold_report.py

- 유형별 개수, render_as 분포, Solution knowledge_type 분포
- world 시스템별 등장 gold 수 (한 번도 안 나온 시스템 목록)
- _rejected 기록 수
- generate_gold.py 호출 로그(data/generated/gold_usage.jsonl)의 총 토큰·비용
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, get_args

from common.schema.gold import GoldKnowledge, ScenarioType, load_gold_knowledge
from common.schema.knowledge import EntityType
from common.schema.world import World, load_world
from gold_rules import system_mentions

ROOT = Path(__file__).resolve().parents[2]
WORLD_FILE = ROOT / "data" / "world" / "world.yaml"
GOLD_DIR = ROOT / "data" / "gold" / "knowledge"
REJECTED_DIR = ROOT / "data" / "gold" / "_rejected"
USAGE_LOG = ROOT / "data" / "generated" / "gold_usage.jsonl"

_TOKEN_FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {f: sum(r.get(f, 0) for r in records) for f in _TOKEN_FIELDS}
    costs = [r.get("cost_usd") for r in records]
    totals["calls"] = len(records)
    totals["dry_run_calls"] = sum(1 for r in records if r.get("dry_run"))
    totals["cost_usd"] = sum(c for c in costs if c is not None)
    totals["calls_without_price"] = sum(1 for c in costs if c is None)
    return totals


def read_usage(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_report(
    world: World, gold: list[GoldKnowledge], usage: list[dict[str, Any]], rejected_count: int
) -> str:
    by_type = Counter(g.scenario_type for g in gold)
    render = Counter(r for g in gold for r in g.render_as)
    ktype = Counter(
        str(e.knowledge_type) for g in gold for e in g.entities if e.type is EntityType.SOLUTION
    )
    systems = system_mentions(gold)

    def table(counter: Counter, keys: list[str] | None = None) -> list[str]:
        keys = keys if keys is not None else sorted(counter)
        return [f"  {k:<20} {counter.get(k, 0)}" for k in keys]

    lines = [f"gold 총 {len(gold)}건 ({gold[0].id}~{gold[-1].id})" if gold else "gold 0건"]
    lines += ["", "[유형별]", *table(by_type, list(get_args(ScenarioType)))]
    lines += ["", "[render_as]", *table(render)]
    lines += ["", "[Solution knowledge_type]", *table(ktype)]
    lines += ["", "[world 시스템별 등장 gold 수]"]
    lines += [f"  {s.canonical_name:<20} {systems.get(s.canonical_name, 0)}" for s in world.systems]
    unused = [s.canonical_name for s in world.systems if s.canonical_name not in systems]
    lines.append(f"  미등장 시스템: {', '.join(unused) if unused else '없음'}")
    lines += ["", f"[거부 기록] data/gold/_rejected/ {rejected_count}건"]

    u = summarize_usage(usage)
    lines += ["", "[토큰·비용] (generate_gold.py 호출 로그 합계)"]
    if not usage:
        lines.append("  호출 로그 없음")
    else:
        lines.append(f"  호출 {u['calls']}회 (dry-run {u['dry_run_calls']}회)")
        lines.append(
            f"  입력 {u['prompt_tokens']:,} (cached {u['cached_tokens']:,}) / "
            f"출력 {u['completion_tokens']:,} (reasoning {u['reasoning_tokens']:,})"
        )
        cost = f"${u['cost_usd']:.4f}"
        if u["calls_without_price"]:
            cost += f" (단가 미설정 호출 {u['calls_without_price']}회는 비용 제외)"
        lines.append(f"  비용 {cost}")
    return "\n".join(lines)


def main() -> int:
    world = load_world(WORLD_FILE)
    gold = load_gold_knowledge(GOLD_DIR)
    rejected = len(list(REJECTED_DIR.glob("*.yaml"))) if REJECTED_DIR.exists() else 0
    print(build_report(world, gold, read_usage(USAGE_LOG), rejected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
