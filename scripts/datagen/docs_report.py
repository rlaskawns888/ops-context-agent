"""합성 원문 요약 리포트.

실행:
    uv run python scripts/datagen/docs_report.py

- source_type별 문서 수
- gold 커버리지: 원문이 1개도 없는 gold, render_as 중 아직 원문이 없는 (gold, 유형)
- 품질: evidence 1차 통과율(첫 시도 응답의 evidence 검사 통과 비율), 재시도 횟수, 거부 건수
- render_docs.py 호출 로그(data/generated/docs_usage.jsonl)의 총 토큰·비용
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common.schema.gold import GoldKnowledge, load_gold_knowledge
from gold_report import read_usage, summarize_usage
from render_docs import (
    GENERATED_DIR,
    GOLD_DIR,
    REJECTED_DIR,
    SOURCE_TYPES,
    USAGE_LOG,
    output_file,
    read_documents,
)


def coverage(
    golds: list[GoldKnowledge], docs: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], list[tuple[str, str]]]:
    """(원문이 하나도 없는 gold id, 원문이 없는 (gold id, render_as) 쌍)."""
    have = {(gid, st) for st, items in docs.items() for d in items for gid in d["gold_ids"]}
    covered = {gid for gid, _ in have}
    no_doc = [g.id for g in golds if g.id not in covered]
    missing = [
        (g.id, st.value) for g in golds for st in g.render_as if (g.id, st.value) not in have
    ]
    return no_doc, missing


def quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = [r for r in records if r.get("attempt") == 0]
    return {
        "jobs": len(first),
        "first_evidence_ok": sum(1 for r in first if r.get("evidence_ok") is True),
        "first_ok": sum(1 for r in first if r.get("ok")),
        "retries": sum(1 for r in records if r.get("attempt", 0) > 0),
        "reason_kinds": Counter(
            reason.split("]")[0].lstrip("[")
            for r in records
            for reason in r.get("reasons", [])
            if reason.startswith("[")
        ),
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0"


def build_report(
    golds: list[GoldKnowledge],
    docs: dict[str, list[dict[str, Any]]],
    usage: list[dict[str, Any]],
    rejected_count: int,
) -> str:
    lines = ["[source_type별 문서 수]"]
    lines += [f"  {st:<10} {len(docs.get(st, []))}" for st in docs]
    lines.append(f"  {'합계':<10} {sum(len(v) for v in docs.values())}")

    no_doc, missing = coverage(golds, docs)
    lines += ["", f"[gold 커버리지] 전체 {len(golds)}건"]
    lines.append(f"  원문이 없는 gold {len(no_doc)}건: {', '.join(no_doc) if no_doc else '없음'}")
    shown = ", ".join(f"{g}/{s}" for g, s in missing[:30])
    more = f" 외 {len(missing) - 30}건" if len(missing) > 30 else ""
    lines.append(f"  아직 없는 (gold, render_as) {len(missing)}건: {shown or '없음'}{more}")

    q = quality(usage)
    lines += ["", "[생성 품질] (docs_usage.jsonl 기준, dry-run 포함)"]
    if not usage:
        lines.append("  호출 로그 없음")
    else:
        lines.append(f"  evidence 1차 통과율 {_pct(q['first_evidence_ok'], q['jobs'])}")
        lines.append(f"  전체 검증 1차 통과율 {_pct(q['first_ok'], q['jobs'])}")
        lines.append(f"  재시도 {q['retries']}회")
        kinds = ", ".join(f"{k} {n}" for k, n in q["reason_kinds"].most_common())
        lines.append(f"  실패 사유 분류: {kinds or '없음'}")
    lines.append(f"  거부 기록 data/generated/_rejected/ {rejected_count}건")

    u = summarize_usage(usage)
    lines += ["", "[토큰·비용] (render_docs.py 호출 로그 합계)"]
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
    golds = load_gold_knowledge(GOLD_DIR)
    docs = {st.value: read_documents(output_file(st, GENERATED_DIR)) for st in SOURCE_TYPES}
    rejected = len(list(Path(REJECTED_DIR).glob("*.json"))) if REJECTED_DIR.exists() else 0
    print(build_report(golds, docs, read_usage(USAGE_LOG), rejected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
