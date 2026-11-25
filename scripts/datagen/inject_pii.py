"""개인정보(PII) 삽입 + 최종 코퍼스·정답 라벨 분리 + 매니페스트.

실행:
    uv sync --all-packages --group scripts
    uv run python scripts/datagen/inject_pii.py            # data/raw, data/labels 생성
    uv run python scripts/datagen/inject_pii.py --check    # 저장하지 않고 검증만

LLM 호출 없음. 같은 seed면 결과가 항상 같다 (manifest.json의 generated_at만 다름).

입력 (읽기 전용, 수정하지 않는다)
- data/generated/{slack,ticket,guide,minutes}.jsonl: gold 원문 + 노이즈 원문 (정답 필드 포함)
- data/generated/users.jsonl: 작성자 PII 값 (전화·메일·사번)
- data/generated/{gold,docs,noise}_usage.jsonl: 생성 단계별 호출 로그 (매니페스트 합산용)
- data/gold/knowledge/*.yaml: evidence 보존 검증용

출력
- data/raw/{slack,ticket,guide,minutes}.jsonl: 적재 입력. 정답·생성 필드를 뺀 원문 (doc_id 순)
- data/raw/manifest.json: 생성 과정 요약 (모델·토큰·비용·입력 해시·프롬프트 해시·규모)
- data/labels/doc_gold.jsonl: {doc_id, gold_ids, related_gold_ids, noise_kind, issue_category}
- data/labels/pii.jsonl: {doc_id, field_path, pii_type, value} — 3단계 마스킹 검증 정답

PII 삽입 규칙
- 대상: slack 40%, ticket 60% 문서 (guide·minutes 없음). 문서당 1~2종.
- 값: 문서 요청자(slack은 첫 메시지 작성자, ticket은 requester_user_id)의 값.
  phone·email·employee_no는 users.jsonl, internal_ip·pc_name은 사용자별 seed 고정값.
- 위치: 템플릿 문장을 기존 텍스트 **끝에 덧붙이거나** 요청자 메시지를 새로 추가한다.
  기존 텍스트는 앞부분이 그대로 남으므로 evidence 문장은 바뀌지 않는다.
  - slack: 요청자 메시지 1개 끝에 추가, 또는 그 메시지 바로 뒤(+1분)에 요청자 메시지 1개 추가
  - ticket: description 또는 요청자 댓글 1개 끝에 추가

검증 (저장 전, 실패하면 저장하지 않는다)
- 문서 수 = slack 300, ticket 500, guide 40, minutes 20
- data/raw 레코드에 정답·생성 필드(LABEL_FIELDS)가 없고, 허용 필드만 있음
- 원문 텍스트는 모두 앞부분 그대로 보존, gold evidence는 삽입 후에도 그대로 존재
- pii.jsonl의 value가 해당 doc_id의 field_path 텍스트에 실제로 존재
- PII 삽입 후 문서가 문서 모델(common.schema.documents) 검증을 통과
- 실행 전후로 data/generated 입력 파일이 바뀌지 않음
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import generate_gold
import render_docs
import render_noise
from common.schema.documents import DOCUMENT_MODELS
from common.schema.gold import GoldKnowledge, load_gold_knowledge
from common.schema.knowledge import SourceType
from common.schema.users import MockUser
from docs_report import TARGET_COUNTS
from gold_report import read_usage, summarize_usage
from render_docs import (
    GENERATED_DIR,
    GOLD_DIR,
    SOURCE_TYPES,
    USERS_FILE,
    WORLD_FILE,
    load_users,
    output_file,
    read_documents,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LABELS_DIR = ROOT / "data" / "labels"
MANIFEST_FILE = "manifest.json"
DEFAULT_SEED = 42

PII_RATIO: dict[str, float] = {"slack": 0.4, "ticket": 0.6, "guide": 0.0, "minutes": 0.0}
PII_TYPES: tuple[str, ...] = ("phone", "email", "employee_no", "internal_ip", "pc_name")
MAX_PII_TYPES_PER_DOC = 2
SLACK_MESSAGES_MAX = 10  # common.schema.documents.SLACK_MESSAGES_MAX와 같다 (새 메시지 추가 한도)
NEW_MESSAGE_GAP = timedelta(minutes=1)

# 값 바로 뒤에 조사(로/으로)를 붙이지 않는다 — 숫자 끝자리에 따라 맞는 조사가 달라진다.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "phone": (
        "연락은 {v} 번호로 부탁드려요.",
        "급하시면 휴대폰({v})으로 연락 주세요.",
        "제 휴대폰 번호는 {v}입니다.",
        "통화 필요하시면 이 번호로 주세요: {v}",
        "자리에 없으면 문자 남겨주세요. 번호는 {v}입니다.",
    ),
    "email": (
        "메일은 {v}로 보내주시면 됩니다.",
        "결과는 {v}로 회신 부탁드립니다.",
        "제 메일 주소는 {v}입니다.",
        "관련 자료는 {v}로 공유 부탁드려요.",
        "처리되면 메일({v})로 알려주세요.",
    ),
    "employee_no": (
        "사번 {v}입니다.",
        "제 사번은 {v}입니다.",
        "확인용으로 사번 남깁니다: {v}",
        "사번({v})으로 조회해 보시면 됩니다.",
        "본인 확인 필요하시면 사번은 {v}입니다.",
    ),
    "internal_ip": (
        "제 PC IP는 {v}입니다.",
        "ipconfig로 확인한 IP는 {v}입니다.",
        "현재 할당된 IP 주소: {v}",
        "접속한 PC의 내부 IP는 {v}입니다.",
        "IP 필요하실까 봐 남깁니다. {v}입니다.",
    ),
    "pc_name": (
        "PC 이름은 {v}입니다.",
        "사용 중인 PC명: {v}",
        "원격 지원 필요하시면 제 PC({v})로 접속해 주세요.",
        "제 PC명은 {v}입니다.",
        "장비 확인용으로 PC 이름 남깁니다: {v}",
    ),
}

# 생성·정답 관련 필드 — data/raw에서 빼고 data/labels/doc_gold.jsonl로만 옮긴다
LABEL_FIELDS: tuple[str, ...] = (
    "gold_ids",
    "related_gold_ids",
    "issue_category",
    "noise_kind",
    "plan_id",
)
DOC_GOLD_FIELDS: tuple[str, ...] = (
    "doc_id",
    "gold_ids",
    "related_gold_ids",
    "noise_kind",
    "issue_category",
)
ALLOWED_RAW_FIELDS: dict[str, frozenset[str]] = {
    st.value: frozenset(DOCUMENT_MODELS[st].model_fields) - set(LABEL_FIELDS) for st in SOURCE_TYPES
}

USAGE_LOGS: dict[str, Path] = {
    "gold": generate_gold.USAGE_LOG,
    "render_docs": render_docs.USAGE_LOG,
    "render_noise": render_noise.USAGE_LOG,
}
REASONING_EFFORTS: dict[str, str] = {
    "gold": generate_gold.REASONING_EFFORT,
    "render_docs": render_docs.REASONING_EFFORT,
    "render_noise": render_noise.REASONING_EFFORT,
}
# 스크립트별 프롬프트를 만드는 고정 템플릿 (문자열·dict는 값, 함수는 소스 코드를 해시한다)
PROMPT_COMPONENTS: dict[str, tuple[Any, tuple[str, ...]]] = {
    "scripts/datagen/generate_gold.py": (
        generate_gold,
        ("_COMMON_RULES", "_TYPE_RULES", "build_messages"),
    ),
    "scripts/datagen/render_docs.py": (
        render_docs,
        ("_SYSTEM_RULES", "_STYLE", "_ROLE_NOTE", "build_messages", "_date_section"),
    ),
    "scripts/datagen/render_noise.py": (
        render_noise,
        (
            "_SYSTEM_RULES",
            "_STYLE",
            "build_messages",
            "situation_block",
            "_outcome",
            "_date_line",
        ),
    ),
}
CATEGORY_RATIO_SOURCE = "assumed (not real statistics)"

Doc = dict[str, Any]


@dataclass(frozen=True)
class PiiLabel:
    doc_id: str
    field_path: str
    pii_type: str
    value: str


@dataclass
class Corpus:
    raw: dict[str, list[Doc]]  # source_type → data/raw 레코드
    labeled: dict[str, list[Doc]]  # PII 삽입 후, 정답 필드를 빼기 전 (검증용)
    doc_gold: list[Doc]
    pii: list[PiiLabel]


# ---------------------------------------------------------------------------
# 사용자별 PII 값
# ---------------------------------------------------------------------------


def user_pii_profiles(users: Sequence[MockUser], seed: int) -> dict[str, dict[str, str]]:
    """user_id → {pii_type: value}. internal_ip·pc_name은 seed로 사용자마다 고정, 중복 없음."""
    rng = random.Random(f"{seed}:user-pii")
    used: set[str] = set()

    def unique(make: Callable[[], str]) -> str:
        value = make()
        while value in used:
            value = make()
        used.add(value)
        return value

    out: dict[str, dict[str, str]] = {}
    for u in sorted(users, key=lambda u: u.user_id):
        out[u.user_id] = {
            "phone": u.phone,
            "email": u.email,
            "employee_no": u.employee_no,
            "internal_ip": unique(
                lambda: f"10.{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}"
            ),
            "pc_name": unique(lambda: f"PC-{rng.randint(1_000_000, 9_999_999)}"),
        }
    return out


# ---------------------------------------------------------------------------
# 필드 경로 (예: "messages[3].text", "description")
# ---------------------------------------------------------------------------

_PATH_PART = re.compile(r"([A-Za-z_]+)(?:\[(\d+)\])?")


def get_field(doc: Doc, path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        m = _PATH_PART.fullmatch(part)
        if m is None:
            raise ValueError(f"잘못된 field_path: {path!r}")
        node = node[m.group(1)]
        if m.group(2) is not None:
            node = node[int(m.group(2))]
    return node


def text_fields(doc: Doc) -> Iterator[tuple[str, str]]:
    """(field_path, 텍스트) — 문서 모델의 text_fields()와 같은 범위."""
    yield "title", doc["title"]
    st = doc["source_type"]
    if st == "slack":
        for i, m in enumerate(doc["messages"]):
            yield f"messages[{i}].text", m["text"]
    elif st == "ticket":
        yield "description", doc["description"]
        for i, c in enumerate(doc["comments"]):
            yield f"comments[{i}].text", c["text"]
        yield "resolution", doc["resolution"]
    elif st == "guide":
        yield "body", doc["body"]
    elif st == "minutes":
        for i, a in enumerate(doc["agenda"]):
            yield f"agenda[{i}]", a
        yield "discussion", doc["discussion"]
        for i, d in enumerate(doc["decisions"]):
            yield f"decisions[{i}]", d
        for i, a in enumerate(doc["action_items"]):
            yield f"action_items[{i}].task", a["task"]


# ---------------------------------------------------------------------------
# 삽입
# ---------------------------------------------------------------------------


def _append(text: str, sentence: str) -> str:
    """기존 텍스트는 그대로 두고 끝에 문장을 붙인다."""
    sep = "" if not text or text[-1].isspace() else " "
    return f"{text}{sep}{sentence}"


def requester_of(doc: Doc, slack_to_user: dict[str, str]) -> str:
    if doc["source_type"] == "ticket":
        return doc["requester_user_id"]
    return slack_to_user[doc["messages"][0]["slack_user_id"]]


def inject_document(
    doc: Doc, requester: MockUser, profile: dict[str, str], rng: random.Random
) -> tuple[Doc, list[PiiLabel]]:
    """doc 사본에 PII 문장 1~2개를 넣고 (새 문서, 라벨)을 돌려준다. 같은 위치에 함께 넣는다."""
    doc = deepcopy(doc)
    k = rng.randint(1, MAX_PII_TYPES_PER_DOC)
    types = rng.sample(PII_TYPES, k)
    picks = [(t, profile[t], rng.choice(TEMPLATES[t]).format(v=profile[t])) for t in types]
    sentence = picks[0][2]
    for _, _, s in picks[1:]:  # "PC명: PC-123" 처럼 마침표 없이 끝나면 쉼표로 잇는다
        sentence += (" " if sentence.endswith(".") else ", ") + s

    if doc["source_type"] == "slack":
        msgs = doc["messages"]
        mine = [i for i, m in enumerate(msgs) if m["slack_user_id"] == requester.slack_user_id]
        i = rng.choice(mine)
        ts = datetime.fromisoformat(msgs[i]["ts"]) + NEW_MESSAGE_GAP
        can_add = len(msgs) < SLACK_MESSAGES_MAX and (
            i == len(msgs) - 1 or datetime.fromisoformat(msgs[i + 1]["ts"]) >= ts
        )
        if can_add and rng.random() < 0.5:
            new = {"slack_user_id": requester.slack_user_id, "ts": ts.isoformat(), "text": sentence}
            msgs.insert(i + 1, new)
            path = f"messages[{i + 1}].text"
        else:
            msgs[i]["text"] = _append(msgs[i]["text"], sentence)
            path = f"messages[{i}].text"
    elif doc["source_type"] == "ticket":
        targets = ["description"] + [
            f"comments[{i}].text"
            for i, c in enumerate(doc["comments"])
            if c["user_id"] == requester.user_id
        ]
        path = rng.choice(targets)
        if path == "description":
            doc["description"] = _append(doc["description"], sentence)
        else:
            c = get_field(doc, path.removesuffix(".text"))
            c["text"] = _append(c["text"], sentence)
    else:
        raise ValueError(f"{doc['source_type']}에는 PII를 넣지 않습니다")

    labels = [PiiLabel(doc["doc_id"], path, t, v) for t, v, _ in picks]
    return doc, labels


def inject_all(
    docs: dict[str, list[Doc]], users: Sequence[MockUser], seed: int
) -> tuple[dict[str, list[Doc]], list[PiiLabel]]:
    """source_type별로 PII_RATIO만큼 문서를 골라 PII를 넣는다 (doc_id 순 정렬된 결과)."""
    by_id = {u.user_id: u for u in users}
    slack_to_user = {u.slack_user_id: u.user_id for u in users}
    profiles = user_pii_profiles(users, seed)
    out: dict[str, list[Doc]] = {}
    labels: list[PiiLabel] = []
    for st, items in docs.items():
        items = deepcopy(sorted(items, key=lambda d: d["doc_id"]))  # 입력과 객체를 공유하지 않음
        rng = random.Random(f"{seed}:pii:{st}")
        n = round(len(items) * PII_RATIO.get(st, 0.0))
        chosen = set(rng.sample([d["doc_id"] for d in items], n))
        result = []
        for d in items:
            if d["doc_id"] in chosen:
                uid = requester_of(d, slack_to_user)
                d, new = inject_document(d, by_id[uid], profiles[uid], rng)
                labels += new
            result.append(d)
        out[st] = result
    return out, labels


# ---------------------------------------------------------------------------
# 코퍼스·라벨 분리
# ---------------------------------------------------------------------------


def split_labels(doc: Doc) -> tuple[Doc, Doc]:
    raw = {k: v for k, v in doc.items() if k not in LABEL_FIELDS}
    label = {
        "doc_id": doc["doc_id"],
        "gold_ids": doc.get("gold_ids", []),
        "related_gold_ids": doc.get("related_gold_ids", []),
        "noise_kind": doc.get("noise_kind"),
        "issue_category": doc.get("issue_category"),
    }
    return raw, label


def build_corpus(docs: dict[str, list[Doc]], users: Sequence[MockUser], seed: int) -> Corpus:
    labeled, pii = inject_all(docs, users, seed)
    raw: dict[str, list[Doc]] = {}
    doc_gold: list[Doc] = []
    for st, items in labeled.items():
        raw[st] = []
        for d in items:
            r, label = split_labels(d)
            raw[st].append(r)
            doc_gold.append(label)
    doc_gold.sort(key=lambda x: x["doc_id"])
    return Corpus(raw=raw, labeled=labeled, doc_gold=doc_gold, pii=pii)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def gold_evidence(gold: GoldKnowledge) -> list[str]:
    items = [*gold.entities, *gold.relations, *gold.policies]
    return list(dict.fromkeys(x.evidence for x in items))


def validate(
    original: dict[str, list[Doc]],
    corpus: Corpus,
    golds: Sequence[GoldKnowledge],
    expected_counts: dict[str, int] | None = TARGET_COUNTS,
) -> list[str]:
    errors: list[str] = []
    gold_by_id = {g.id: g for g in golds}

    if expected_counts is not None:
        for st, n in expected_counts.items():
            got = len(corpus.raw.get(st, []))
            if got != n:
                errors.append(f"[count] {st} 문서 {got}개 (기대 {n})")

    for st, items in corpus.raw.items():
        ids = [d["doc_id"] for d in items]
        if ids != sorted(ids):
            errors.append(f"[order] {st}가 doc_id 순이 아님")
        for d in items:
            leaked = set(d) & set(LABEL_FIELDS)
            extra = set(d) - ALLOWED_RAW_FIELDS[st]
            if leaked:
                errors.append(f"[leak] {d['doc_id']}에 정답 필드 {sorted(leaked)}")
            if extra:
                errors.append(f"[field] {d['doc_id']}에 허용되지 않은 필드 {sorted(extra)}")

    orig_by_id = {d["doc_id"]: d for items in original.values() for d in items}
    for items in corpus.labeled.values():
        for d in items:
            st = SourceType(d["source_type"])
            try:
                DOCUMENT_MODELS[st].model_validate(d)
            except ValueError as e:
                errors.append(f"[model] {d['doc_id']}: {str(e).splitlines()[0]}")
            orig = orig_by_id[d["doc_id"]]
            new_texts = [t for _, t in text_fields(d)]
            # 원문 텍스트 필드는 모두 앞부분 그대로 남아 있어야 한다 (끝에 덧붙이기만 허용)
            remaining = list(new_texts)
            for path, text in text_fields(orig):
                hit = next((t for t in remaining if t.startswith(text)), None)
                if hit is None:
                    errors.append(f"[preserve] {d['doc_id']} {path} 원문이 바뀜")
                else:
                    remaining.remove(hit)
            # gold evidence: 원래 있던 문장은 삽입 후에도 그대로
            orig_joined = "\n".join(t for _, t in text_fields(orig))
            new_joined = "\n".join(new_texts)
            for gid in orig.get("gold_ids", []):
                for ev in gold_evidence(gold_by_id[gid]):
                    if new_joined.count(ev) < orig_joined.count(ev):
                        errors.append(f"[evidence] {d['doc_id']} {gid} 문장 사라짐: {ev[:30]}")

    raw_by_id = {d["doc_id"]: d for items in corpus.raw.values() for d in items}
    for p in corpus.pii:
        doc = raw_by_id.get(p.doc_id)
        if doc is None:
            errors.append(f"[pii] {p.doc_id} 문서 없음")
            continue
        try:
            text = get_field(doc, p.field_path)
        except (KeyError, IndexError, ValueError):
            errors.append(f"[pii] {p.doc_id} {p.field_path} 경로 없음")
            continue
        if not isinstance(text, str) or p.value not in text:
            errors.append(f"[pii] {p.doc_id} {p.field_path}에 {p.pii_type} 값 없음")
    return errors


# ---------------------------------------------------------------------------
# 매니페스트
# ---------------------------------------------------------------------------


def _normalized_bytes(path: Path) -> bytes:
    """텍스트 파일은 CRLF → LF로 맞춰 OS(git autocrlf)와 무관한 해시를 만든다."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_normalized_bytes(path)).hexdigest()


def sha256_dir(path: Path, pattern: str = "*.yaml") -> str:
    h = hashlib.sha256()
    for f in sorted(path.glob(pattern)):
        h.update(f.name.encode() + b"\0" + _normalized_bytes(f) + b"\0")
    return h.hexdigest()


def _component_value(obj: Any) -> Any:
    if callable(obj):
        return inspect.getsource(obj)
    if isinstance(obj, dict):
        return {str(getattr(k, "value", k)): v for k, v in obj.items()}
    return obj


def prompt_template_hashes() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for script, (module, names) in PROMPT_COMPONENTS.items():
        payload = {n: _component_value(getattr(module, n)) for n in names}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        out[script] = {"sha256": hashlib.sha256(blob).hexdigest(), "components": list(names)}
    return out


def _single_or_list(values: set[str]) -> str | list[str] | None:
    if not values:
        return None
    return next(iter(values)) if len(values) == 1 else sorted(values)


def usage_summary(usage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stages = {}
    for stage, records in usage.items():
        s = summarize_usage(records)
        s["cost_usd"] = round(s["cost_usd"], 6)
        stages[stage] = {"usage_log": f"data/generated/{USAGE_LOGS[stage].name}", **s}
    keys = [k for k in next(iter(stages.values())) if k != "usage_log"]
    total = {k: sum(s[k] for s in stages.values()) for k in keys}
    total["cost_usd"] = round(total["cost_usd"], 6)
    return {"stages": stages, "total": total}


def build_manifest(
    corpus: Corpus,
    usage: dict[str, list[dict[str, Any]]],
    seed: int,
    generated_at: datetime,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    models = {r["model"] for records in usage.values() for r in records if r.get("model")}
    efforts = set(REASONING_EFFORTS.values())
    doc_st = {d["doc_id"]: st for st, items in corpus.raw.items() for d in items}
    pii_doc_ids = {p.doc_id for p in corpus.pii}
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "synthetic": True,
        "seed": seed,
        "llm": {
            "chat_deployment": _single_or_list(models),
            "reasoning_effort": _single_or_list(efforts),
            "reasoning_effort_by_stage": REASONING_EFFORTS,
        },
        "usage": usage_summary(usage),
        "usage_note": "usage 로그 전체 합산 (dry-run·재시도 호출 포함). "
        "cost_usd는 단가가 설정된 호출만 합산 (calls_without_price 참고)",
        "inputs_sha256": input_hashes,
        "prompt_templates_sha256": prompt_template_hashes(),
        "hash_note": "텍스트 파일은 CRLF를 LF로 바꾼 뒤 해시, "
        "디렉터리는 파일명 순으로 이름+내용을 해시",
        "documents": {
            **{st: len(items) for st, items in corpus.raw.items()},
            "total": sum(len(items) for items in corpus.raw.values()),
        },
        "pii": {
            "target_ratio": PII_RATIO,
            "documents": dict(sorted(Counter(doc_st[i] for i in pii_doc_ids).items())),
            "documents_total": len(pii_doc_ids),
            "by_type": {t: sum(1 for p in corpus.pii if p.pii_type == t) for t in PII_TYPES},
            "values_total": len(corpus.pii),
        },
        "category_ratio_source": CATEGORY_RATIO_SOURCE,
    }


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------


def _jsonl(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def render_outputs(corpus: Corpus, manifest: dict[str, Any] | None) -> dict[str, str]:
    """상대 경로(data/ 기준 raw/…, labels/…) → 파일 내용."""
    files = {f"raw/{st}.jsonl": _jsonl(items) for st, items in corpus.raw.items()}
    files["labels/doc_gold.jsonl"] = _jsonl(corpus.doc_gold)
    files["labels/pii.jsonl"] = _jsonl([asdict(p) for p in corpus.pii])
    if manifest is not None:
        files[f"raw/{MANIFEST_FILE}"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return files


def input_files(generated_dir: Path) -> list[Path]:
    return [
        *(output_file(st, generated_dir) for st in SOURCE_TYPES),
        generated_dir / USERS_FILE.name,
        *(generated_dir / p.name for p in USAGE_LOGS.values()),
    ]


def snapshot(paths: Sequence[Path]) -> dict[Path, str | None]:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in paths}


def load_inputs(generated_dir: Path) -> tuple[dict[str, list[Doc]], list[MockUser]]:
    docs = {st.value: read_documents(output_file(st, generated_dir)) for st in SOURCE_TYPES}
    users = load_users(generated_dir / USERS_FILE.name)
    return docs, users


def run(
    *,
    seed: int,
    generated_dir: Path,
    gold_dir: Path,
    world_file: Path,
    raw_dir: Path,
    labels_dir: Path,
    write: bool = True,
    generated_at: datetime | None = None,
) -> tuple[Corpus, dict[str, Any], list[str]]:
    paths = input_files(generated_dir)
    before = snapshot(paths)

    docs, users = load_inputs(generated_dir)
    golds = load_gold_knowledge(gold_dir)
    corpus = build_corpus(docs, users, seed)
    errors = validate(docs, corpus, golds)

    usage = {stage: read_usage(generated_dir / p.name) for stage, p in USAGE_LOGS.items()}
    hashes = {
        "data/world/world.yaml": sha256_file(world_file),
        "data/gold/knowledge": sha256_dir(gold_dir),
        "data/generated/users.jsonl": sha256_file(generated_dir / USERS_FILE.name),
    }
    manifest = build_manifest(corpus, usage, seed, generated_at or datetime.now(UTC), hashes)

    if snapshot(paths) != before:
        errors.append("[input] 실행 중 data/generated 입력 파일이 바뀜")

    if write and not errors:
        for rel, content in render_outputs(corpus, manifest).items():
            root = raw_dir if rel.startswith("raw/") else labels_dir
            path = root / rel.split("/", 1)[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    return corpus, manifest, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PII 삽입 + 코퍼스·라벨 분리 + 매니페스트")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true", help="저장하지 않고 검증만")
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    args = parser.parse_args(argv)

    corpus, manifest, errors = run(
        seed=args.seed,
        generated_dir=args.generated_dir,
        gold_dir=GOLD_DIR,
        world_file=WORLD_FILE,
        raw_dir=args.raw_dir,
        labels_dir=args.labels_dir,
        write=not args.check,
    )
    print(f"문서 수: {manifest['documents']}")
    print(f"PII: {manifest['pii']['documents']} / 종류별 {manifest['pii']['by_type']}")
    print(f"총 비용: ${manifest['usage']['total']['cost_usd']:.4f}")
    if errors:
        print(f"검증 실패 {len(errors)}건 — 저장하지 않음", file=sys.stderr)
        for e in errors[:50]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(
        "검증 통과"
        + (" (--check, 저장 안 함)" if args.check else f" → {args.raw_dir}, {args.labels_dir}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
