"""합성 원문 요약 리포트.

실행:
    uv run python scripts/datagen/docs_report.py

gold 원문 (render_docs.py)
- source_type별 gold 원문 수
- gold 커버리지: 원문이 1개도 없는 gold, render_as 중 아직 원문이 없는 (gold, 유형)
- 품질: evidence 1차 통과율(첫 시도 응답의 evidence 검사 통과 비율), 재시도 횟수, 거부 건수
- render_docs.py 호출 로그(data/generated/docs_usage.jsonl)의 총 토큰·비용

노이즈 (render_noise.py)
- source_type별 총 문서 수 (gold 원문 + 노이즈, 목표 대비)
- 노이즈 category·noise_kind·status 분포
- 렌더링 1차 통과율(항목 단위), 재시도, 거부, 토큰·비용 (data/generated/noise_usage.jsonl)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common.schema.gold import GoldKnowledge, load_gold_knowledge
from gold_report import read_usage, summarize_usage
from plan_noise import PLAN_FILE, load_plan
from render_docs import (
    GENERATED_DIR,
    GOLD_DIR,
    REJECTED_DIR,
    SOURCE_TYPES,
    USAGE_LOG,
    output_file,
    read_documents,
)
from render_noise import REJECTED_PREFIX
from render_noise import USAGE_LOG as NOISE_USAGE_LOG

# 목표 규모 (gold 원문 + 노이즈)
TARGET_COUNTS: dict[str, int] = {"slack": 300, "ticket": 500, "guide": 40, "minutes": 20}


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


def noise_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """render_noise 호출 로그 → 항목 단위 1차 통과율·재시도 호출 수·실패 사유 분류."""
    first = [r for r in records if r.get("attempt") == 0]
    return {
        "items": sum(len(r.get("plan_ids", [])) for r in first),
        "first_ok": sum(len(r.get("ok_plan_ids", [])) for r in first),
        "retries": sum(1 for r in records if r.get("attempt", 0) > 0),
        "reason_kinds": Counter(
            reason.split("]")[0].lstrip("[")
            for r in records
            for reasons in r.get("failed", {}).values()
            for reason in reasons
            if reason.startswith("[")
        ),
    }


def is_noise(doc: dict[str, Any]) -> bool:
    return bool(doc.get("noise_kind"))


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0"


def _dist(counter: Counter) -> str:
    return ", ".join(f"{k} {n}" for k, n in counter.most_common()) or "없음"


def _usage_lines(usage: list[dict[str, Any]], title: str) -> list[str]:
    lines = ["", f"[토큰·비용] ({title})"]
    if not usage:
        return [*lines, "  호출 로그 없음"]
    u = summarize_usage(usage)
    lines.append(f"  호출 {u['calls']}회 (dry-run {u['dry_run_calls']}회)")
    lines.append(
        f"  입력 {u['prompt_tokens']:,} (cached {u['cached_tokens']:,}) / "
        f"출력 {u['completion_tokens']:,} (reasoning {u['reasoning_tokens']:,})"
    )
    cost = f"${u['cost_usd']:.4f}"
    if u["calls_without_price"]:
        cost += f" (단가 미설정 호출 {u['calls_without_price']}회는 비용 제외)"
    lines.append(f"  비용 {cost}")
    return lines


def build_report(
    golds: list[GoldKnowledge],
    docs: dict[str, list[dict[str, Any]]],
    usage: list[dict[str, Any]],
    rejected_count: int,
) -> str:
    """gold 원문 리포트 (노이즈 문서는 제외하고 센다)."""
    docs = {st: [d for d in items if not is_noise(d)] for st, items in docs.items()}
    lines = ["[source_type별 gold 원문 수]"]
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
        lines.append(f"  실패 사유 분류: {_dist(q['reason_kinds'])}")
    lines.append(f"  거부 기록 data/generated/_rejected/ {rejected_count}건")
    lines += _usage_lines(usage, "render_docs.py 호출 로그 합계")
    return "\n".join(lines)


def build_noise_report(
    docs: dict[str, list[dict[str, Any]]],
    usage: list[dict[str, Any]],
    rejected_count: int,
    plan_counts: dict[str, int] | None = None,
) -> str:
    lines = ["[source_type별 총 문서 수] (gold 원문 + 노이즈 / 목표)"]
    for st, target in TARGET_COUNTS.items():
        items = docs.get(st, [])
        noise = sum(1 for d in items if is_noise(d))
        planned = f", 노이즈 계획 {plan_counts.get(st, 0)}" if plan_counts is not None else ""
        lines.append(
            f"  {st:<10} {len(items)} / {target}  "
            f"(gold {len(items) - noise}, 노이즈 {noise}{planned})"
        )
    total = sum(len(docs.get(st, [])) for st in TARGET_COUNTS)
    lines.append(f"  {'합계':<10} {total} / {sum(TARGET_COUNTS.values())}")

    lines += ["", "[노이즈 분포]"]
    shown = False
    for st in TARGET_COUNTS:
        noise = [d for d in docs.get(st, []) if is_noise(d)]
        if not noise:
            continue
        shown = True
        lines.append(f"  {st} ({len(noise)}건)")
        lines.append(f"    category   {_dist(Counter(d.get('issue_category') for d in noise))}")
        lines.append(f"    noise_kind {_dist(Counter(d['noise_kind'] for d in noise))}")
        if st == "ticket":
            lines.append(f"    status     {_dist(Counter(d.get('status') for d in noise))}")
        elif st == "slack":  # slack 문서에는 status 필드가 없어 미해결 여부만 센다
            unresolved = sum(1 for d in noise if d["noise_kind"] == "unresolved")
            lines.append(f"    status     미해결 {unresolved}, 답변 완료 {len(noise) - unresolved}")
    if not shown:
        lines.append("  노이즈 문서 없음")

    q = noise_quality(usage)
    lines += ["", "[노이즈 렌더링 품질] (noise_usage.jsonl 기준, dry-run 포함)"]
    if not usage:
        lines.append("  호출 로그 없음")
    else:
        lines.append(f"  1차 통과율 (항목 단위) {_pct(q['first_ok'], q['items'])}")
        lines.append(f"  재시도 호출 {q['retries']}회")
        lines.append(f"  실패 사유 분류: {_dist(q['reason_kinds'])}")
    lines.append(f"  거부 기록 data/generated/_rejected/{REJECTED_PREFIX}* {rejected_count}건")
    lines += _usage_lines(usage, "render_noise.py 호출 로그 합계")
    return "\n".join(lines)


def main() -> int:
    golds = load_gold_knowledge(GOLD_DIR)
    docs = {st.value: read_documents(output_file(st, GENERATED_DIR)) for st in SOURCE_TYPES}
    files = list(Path(REJECTED_DIR).glob("*.json")) if REJECTED_DIR.exists() else []
    noise_rejected = sum(1 for f in files if f.name.startswith(REJECTED_PREFIX))
    print(build_report(golds, docs, read_usage(USAGE_LOG), len(files) - noise_rejected))

    plan_counts = None
    if PLAN_FILE.exists():
        plan_counts = dict(Counter(i.source_type.value for i in load_plan(PLAN_FILE)))
    print("\n========== 노이즈 ==========\n")
    print(build_noise_report(docs, read_usage(NOISE_USAGE_LOG), noise_rejected, plan_counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
