"""합성 원문 스키마: 노이즈 필드 추가 후에도 기존 gold 원문이 그대로 로드되는지."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from common.schema.documents import Document, SlackThread

ROOT = Path(__file__).resolve().parents[3]
GENERATED = ROOT / "data" / "generated"
DOC_FILES = [GENERATED / f"{st}.jsonl" for st in ("slack", "ticket", "guide", "minutes")]
ADAPTER = TypeAdapter(Document)


def _slack(**overrides):
    base = {
        "doc_id": "SL-0001",
        "source_type": "slack",
        "created_at": "2025-03-04T10:00:00",
        "title": "VPN 문의",
        "gold_ids": ["K001"],
        "author_user_ids": ["U0001"],
        "channel": "#it-helpdesk",
        "messages": [
            {"slack_user_id": "S0000000A", "ts": f"2025-03-04T10:0{i}:00", "text": "문의"}
            for i in range(3)
        ],
    }
    return {**base, **overrides}


def test_gold_document_without_new_fields_loads_with_defaults():
    doc = SlackThread.model_validate(_slack())
    assert doc.related_gold_ids == [] and doc.noise_kind is None
    assert doc.issue_category is None and doc.plan_id is None


def test_gold_document_still_requires_gold_ids():
    with pytest.raises(ValidationError, match="gold_ids가 1개 이상"):
        SlackThread.model_validate(_slack(gold_ids=[]))


def test_noise_document_rules():
    noise = {"gold_ids": [], "noise_kind": "general", "plan_id": "NS-0001", "issue_category": "VPN"}
    assert SlackThread.model_validate(_slack(**noise)).gold_ids == []
    with pytest.raises(ValidationError, match="gold_ids가 비어"):
        SlackThread.model_validate(_slack(**{**noise, "gold_ids": ["K001"]}))
    with pytest.raises(ValidationError, match="plan_id가 필요"):
        SlackThread.model_validate(_slack(**{**noise, "plan_id": None}))
    with pytest.raises(ValidationError, match="related_gold_ids"):
        SlackThread.model_validate(_slack(**{**noise, "noise_kind": "repeat"}))
    with pytest.raises(ValidationError, match="related_gold_ids"):
        SlackThread.model_validate(_slack(**{**noise, "related_gold_ids": ["K001"]}))
    repeat = {**noise, "noise_kind": "repeat", "related_gold_ids": ["K001"]}
    assert SlackThread.model_validate(_slack(**repeat)).related_gold_ids == ["K001"]


@pytest.mark.skipif(
    not all(p.exists() for p in DOC_FILES), reason="data/generated 원문이 없음 (gitignore)"
)
def test_existing_generated_documents_load():
    gold_docs = 0
    for path in DOC_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = ADAPTER.validate_python(json.loads(line))
            if doc.noise_kind is None:
                gold_docs += 1
                assert doc.gold_ids
    assert gold_docs >= 105
