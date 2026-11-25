"""PII 삽입·코퍼스/라벨 분리·매니페스트 (LLM 호출 없음)."""

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

import inject_pii as ip
import render_docs as rd
from common.schema.gold import load_gold_knowledge
from common.schema.users import MockUser

ROOT = Path(__file__).resolve().parents[2]
GOLDS = load_gold_knowledge(rd.GOLD_DIR)
GOLD_BY_ID = {g.id: g for g in GOLDS}
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)

HAS_GENERATED = all(p.exists() for p in ip.input_files(rd.GENERATED_DIR)[:5])
needs_generated = pytest.mark.skipif(
    not HAS_GENERATED, reason="data/generated 원문이 없음 (gitignore)"
)

# ---------------------------------------------------------------------------
# 작은 합성 입력
# ---------------------------------------------------------------------------


def _user(n: int, slack: str) -> MockUser:
    return MockUser(
        user_id=f"U{n:04d}",
        employee_no=f"20200{n:02d}",
        name=f"사용자{n}",
        email=f"user{n}@opsctx.example",
        phone=f"010-1234-{n:04d}",
        slack_user_id=slack,
        department="영업",
        user_group="정규직",
        role="requester",
        it_team=None,
        hire_date="2020-01-01",
        account_status="ACTIVE",
        granted_permissions=[],
    )


REQ = _user(1, "SREQ00001")
STAFF = _user(2, "SSTF00002")
USERS = [REQ, STAFF]
EVIDENCE = "VPN에 접속하려고 하면 E-102 오류가 뜨면서 연결이 끊깁니다."


def _slack(doc_id: str, n_messages: int = 4, gap_min: int = 10) -> dict:
    msgs = [
        {
            "slack_user_id": (REQ if i % 2 == 0 else STAFF).slack_user_id,
            "ts": f"2026-03-02T09:{i * gap_min:02d}:00",
            "text": EVIDENCE if i == 0 else f"메시지 {i}",
        }
        for i in range(n_messages)
    ]
    return {
        "doc_id": doc_id,
        "source_type": "slack",
        "created_at": msgs[0]["ts"],
        "title": "VPN 문의",
        "gold_ids": ["K001"],
        "author_user_ids": [REQ.user_id, STAFF.user_id],
        "channel": "#it-helpdesk",
        "messages": msgs,
    }


def _ticket(doc_id: str, noise: bool = False) -> dict:
    doc = {
        "doc_id": doc_id,
        "source_type": "ticket",
        "created_at": "2026-03-02T09:00:00",
        "title": "VPN 오류",
        "gold_ids": [] if noise else ["K001"],
        "author_user_ids": [REQ.user_id, STAFF.user_id],
        "ticket_no": "SR-1",
        "category": "장애",
        "status": "RESOLVED",
        "requester_user_id": REQ.user_id,
        "assignee_user_id": STAFF.user_id,
        "resolved_at": "2026-03-02T10:00:00",
        "description": EVIDENCE,
        "comments": [
            {"user_id": STAFF.user_id, "created_at": "2026-03-02T09:10:00", "text": "재부팅"},
            {"user_id": REQ.user_id, "created_at": "2026-03-02T09:20:00", "text": "됩니다"},
        ],
        "resolution": "재부팅으로 해결",
    }
    if noise:
        doc |= {
            "related_gold_ids": [],
            "issue_category": "VPN",
            "noise_kind": "general",
            "plan_id": "NT-0001",
        }
    return doc


def _docs() -> dict[str, list[dict]]:
    return {
        "slack": [_slack(f"SL-{i:04d}") for i in range(1, 11)],
        "ticket": [_ticket(f"TK-{i:04d}", noise=i % 2 == 0) for i in range(1, 11)],
        "guide": [],
        "minutes": [],
    }


def _profile() -> dict[str, str]:
    return ip.user_pii_profiles(USERS, 42)[REQ.user_id]


# ---------------------------------------------------------------------------
# 사용자 PII 값·삽입
# ---------------------------------------------------------------------------


def test_profiles_fixed_by_seed_and_unique():
    a = ip.user_pii_profiles(USERS, 42)
    assert a == ip.user_pii_profiles(list(reversed(USERS)), 42)
    assert a != ip.user_pii_profiles(USERS, 7)
    p = a[REQ.user_id]
    assert (p["phone"], p["email"], p["employee_no"]) == (REQ.phone, REQ.email, REQ.employee_no)
    assert p["internal_ip"].startswith("10.")
    assert p["pc_name"].startswith("PC-")
    assert a[REQ.user_id]["internal_ip"] != a[STAFF.user_id]["internal_ip"]


def test_templates_at_least_five_per_type():
    assert set(ip.TEMPLATES) == set(ip.PII_TYPES)
    for t, items in ip.TEMPLATES.items():
        assert len(set(items)) >= 5, t
        assert all("{v}" in s for s in items)


@pytest.mark.parametrize("seed", range(30))
def test_slack_injection_appends_or_adds_requester_message(seed):
    doc = _slack("SL-0001")
    new, labels = ip.inject_document(doc, REQ, _profile(), random.Random(seed))
    assert doc == _slack("SL-0001")  # 입력은 그대로
    assert 1 <= len(labels) <= 2 and len({x.pii_type for x in labels}) == len(labels)
    assert len({x.field_path for x in labels}) == 1
    target = ip.get_field(new, labels[0].field_path)
    assert all(x.value in target for x in labels)
    idx = int(labels[0].field_path.split("[")[1].split("]")[0])
    assert new["messages"][idx]["slack_user_id"] == REQ.slack_user_id
    # 원래 메시지는 앞부분이 그대로 남는다 (새 메시지면 원래 메시지들이 순서대로 그대로)
    if len(new["messages"]) == len(doc["messages"]):
        for old, cur in zip(doc["messages"], new["messages"], strict=True):
            assert cur["text"].startswith(old["text"])
    else:
        rest = new["messages"][:idx] + new["messages"][idx + 1 :]
        assert rest == doc["messages"]
    ts = [m["ts"] for m in new["messages"]]
    assert ts == sorted(ts)


def test_slack_never_exceeds_message_limit():
    doc = _slack("SL-0001", n_messages=10, gap_min=5)
    for seed in range(30):
        new, _ = ip.inject_document(doc, REQ, _profile(), random.Random(seed))
        assert len(new["messages"]) == 10


@pytest.mark.parametrize("seed", range(30))
def test_ticket_injection_targets_description_or_requester_comment(seed):
    doc = _ticket("TK-0001")
    new, labels = ip.inject_document(doc, REQ, _profile(), random.Random(seed))
    path = labels[0].field_path
    assert path in {"description", "comments[1].text"}  # comments[0]은 담당자
    assert new["description"].startswith(doc["description"])
    assert new["comments"][1]["text"].startswith(doc["comments"][1]["text"])
    assert new["comments"][0] == doc["comments"][0]
    assert all(x.value in ip.get_field(new, path) for x in labels)


def test_guide_and_minutes_rejected():
    with pytest.raises(ValueError):
        ip.inject_document({"doc_id": "GD-001", "source_type": "guide"}, REQ, _profile(), random)


# ---------------------------------------------------------------------------
# 코퍼스·라벨 분리 + 검증
# ---------------------------------------------------------------------------


def test_build_corpus_ratio_split_and_sort():
    docs = _docs()
    docs["slack"].reverse()
    corpus = ip.build_corpus(docs, USERS, 42)
    pii_docs = {p.doc_id for p in corpus.pii}
    assert sum(d.startswith("SL") for d in pii_docs) == 4  # 10 × 40%
    assert sum(d.startswith("TK") for d in pii_docs) == 6  # 10 × 60%
    for st, items in corpus.raw.items():
        assert [d["doc_id"] for d in items] == sorted(d["doc_id"] for d in items)
        for d in items:
            assert not set(d) & set(ip.LABEL_FIELDS)
            assert set(d) <= ip.ALLOWED_RAW_FIELDS[st]
    tk2 = next(x for x in corpus.doc_gold if x["doc_id"] == "TK-0002")
    assert tk2 == {
        "doc_id": "TK-0002",
        "gold_ids": [],
        "related_gold_ids": [],
        "noise_kind": "general",
        "issue_category": "VPN",
    }
    assert ip.validate(docs, corpus, GOLDS, expected_counts=None) == []


def test_build_corpus_deterministic():
    a = ip.build_corpus(_docs(), USERS, 42)
    b = ip.build_corpus(_docs(), USERS, 42)
    assert ip.render_outputs(a, None) == ip.render_outputs(b, None)
    c = ip.build_corpus(_docs(), USERS, 43)
    assert ip.render_outputs(a, None) != ip.render_outputs(c, None)


def test_validate_catches_leak_evidence_and_pii_mismatch():
    docs = _docs()
    corpus = ip.build_corpus(docs, USERS, 42)
    corpus.raw["slack"][0]["gold_ids"] = ["K001"]
    labeled = next(d for d in corpus.labeled["ticket"] if d["doc_id"] == "TK-0001")
    labeled["description"] = labeled["description"].replace("E-102", "E-103")
    p = corpus.pii[0]
    corpus.pii[0] = ip.PiiLabel(p.doc_id, "title", p.pii_type, p.value)
    kinds = {e.split("]")[0] + "]" for e in ip.validate(docs, corpus, GOLDS, None)}
    assert {"[leak]", "[field]", "[preserve]", "[evidence]", "[pii]"} <= kinds


def test_validate_counts():
    docs = _docs()
    corpus = ip.build_corpus(docs, USERS, 42)
    errors = ip.validate(docs, corpus, GOLDS, {"slack": 10, "ticket": 11})
    assert errors == ["[count] ticket 문서 10개 (기대 11)"]


def test_manifest_fields():
    corpus = ip.build_corpus(_docs(), USERS, 42)
    usage = {
        stage: [
            {
                "model": "m",
                "dry_run": False,
                "prompt_tokens": 10,
                "cached_tokens": 1,
                "completion_tokens": 5,
                "reasoning_tokens": 2,
                "cost_usd": 0.5,
            },
        ]
        for stage in ip.USAGE_LOGS
    }
    m = ip.build_manifest(corpus, usage, 42, FIXED_TIME, {"x": "y"})
    assert m["llm"]["chat_deployment"] == "m"
    assert m["usage"]["total"]["calls"] == 3 and m["usage"]["total"]["cost_usd"] == 1.5
    assert m["usage"]["stages"]["render_noise"]["prompt_tokens"] == 10
    assert m["documents"] == {"slack": 10, "ticket": 10, "guide": 0, "minutes": 0, "total": 20}
    assert m["pii"]["documents"] == {"slack": 4, "ticket": 6}
    assert sum(m["pii"]["by_type"].values()) == m["pii"]["values_total"] == len(corpus.pii)
    assert m["category_ratio_source"] == "assumed (not real statistics)"
    assert set(m["prompt_templates_sha256"]) == set(ip.PROMPT_COMPONENTS)


# ---------------------------------------------------------------------------
# 실제 데이터 (data/generated → data/raw·labels)
# ---------------------------------------------------------------------------


def _run(tmp: Path, seed: int = 42, when: datetime = FIXED_TIME):
    return ip.run(
        seed=seed,
        generated_dir=rd.GENERATED_DIR,
        gold_dir=rd.GOLD_DIR,
        world_file=rd.WORLD_FILE,
        raw_dir=tmp / "raw",
        labels_dir=tmp / "labels",
        generated_at=when,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.*"))}


@needs_generated
def test_real_run_twice_same_output_and_inputs_untouched(tmp_path):
    inputs = ip.input_files(rd.GENERATED_DIR)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    _, _, errors = _run(tmp_path / "a")
    assert errors == []
    _run(tmp_path / "b", when=datetime(2030, 1, 1, tzinfo=UTC))
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    manifest_a = json.loads(a.pop(str(Path("raw/manifest.json"))))
    manifest_b = json.loads(b.pop(str(Path("raw/manifest.json"))))
    assert a == b
    assert manifest_a.pop("generated_at") != manifest_b.pop("generated_at")
    assert manifest_a == manifest_b
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs} == before


@needs_generated
def test_real_run_matches_committed_corpus(tmp_path):
    """커밋된 data/raw·labels가 현재 입력과 seed로 재현된다."""
    _run(tmp_path)
    produced = _tree(tmp_path)
    produced.pop(str(Path("raw/manifest.json")))
    for rel, content in produced.items():
        committed = ROOT / "data" / rel
        assert committed.read_bytes().replace(b"\r\n", b"\n") == content, rel


# ---------------------------------------------------------------------------
# 커밋된 최종 코퍼스 (data/raw, data/labels) — data/generated 없이도 검사
# ---------------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


RAW = {st: _read(ip.RAW_DIR / f"{st}.jsonl") for st in ip.TARGET_COUNTS}
RAW_BY_ID = {d["doc_id"]: d for items in RAW.values() for d in items}
DOC_GOLD = {x["doc_id"]: x for x in _read(ip.LABELS_DIR / "doc_gold.jsonl")}
PII = _read(ip.LABELS_DIR / "pii.jsonl")


def test_committed_counts_and_order():
    assert {st: len(v) for st, v in RAW.items()} == ip.TARGET_COUNTS
    for items in RAW.values():
        ids = [d["doc_id"] for d in items]
        assert ids == sorted(ids)
    assert set(DOC_GOLD) == set(RAW_BY_ID)


def test_committed_raw_has_no_label_fields():
    for st, items in RAW.items():
        for d in items:
            assert not set(d) & set(ip.LABEL_FIELDS), d["doc_id"]
            assert set(d) <= ip.ALLOWED_RAW_FIELDS[st], d["doc_id"]


def test_committed_gold_evidence_present():
    """gold 원문은 (개정 전·후 역할 중 하나의) 필수 evidence를 전부 그대로 담고 있다."""
    pairs = 0
    for doc_id, label in DOC_GOLD.items():
        text = "\n".join(t for _, t in ip.text_fields(RAW_BY_ID[doc_id]))
        for gid in label["gold_ids"]:
            pairs += 1
            assert any(
                all(e in text for e in rd.required_evidence(GOLD_BY_ID[gid], role))
                for role in ("full", "old", "new")
            ), (doc_id, gid)
    assert pairs > 0


def test_committed_pii_positions():
    assert PII
    for p in PII:
        assert set(p) == {"doc_id", "field_path", "pii_type", "value"}
        assert p["pii_type"] in ip.PII_TYPES
        assert p["value"] in ip.get_field(RAW_BY_ID[p["doc_id"]], p["field_path"]), p
    by_type = {
        st: {p["doc_id"] for p in PII if p["doc_id"].startswith(pre)}
        for st, pre in (("slack", "SL"), ("ticket", "TK"), ("guide", "GD"), ("minutes", "MN"))
    }
    assert {st: len(v) for st, v in by_type.items()} == {
        "slack": 120,
        "ticket": 300,
        "guide": 0,
        "minutes": 0,
    }


def test_committed_manifest():
    m = json.loads((ip.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert m["seed"] == ip.DEFAULT_SEED
    assert m["documents"] == {**ip.TARGET_COUNTS, "total": sum(ip.TARGET_COUNTS.values())}
    assert m["pii"]["values_total"] == len(PII)
    assert m["category_ratio_source"] == "assumed (not real statistics)"
    assert m["inputs_sha256"]["data/world/world.yaml"] == ip.sha256_file(rd.WORLD_FILE)
    assert m["inputs_sha256"]["data/gold/knowledge"] == ip.sha256_dir(rd.GOLD_DIR)
    assert m["prompt_templates_sha256"] == ip.prompt_template_hashes()
