"""합성 원문(documents).

gold 지식(gold.py)을 render_as에 적힌 원천 유형으로 풀어 쓴 문서. 3단계 적재·추출의 입력이다.
scripts/datagen/render_docs.py가 생성해
data/generated/{slack,ticket,guide,minutes}.jsonl에 저장한다.

- doc_id 형식: SL-0001(slack), TK-0001(ticket), GD-001(guide), MN-001(minutes)
- gold_ids는 이 원문이 담고 있는 gold id 목록이다. **평가용 숨김 라벨**이라 적재·추출 단계에서
  LLM 입력에 넣지 않는다.
- author_user_ids는 원문에 등장하는 사용자(users.jsonl의 user_id)다.
- 날짜는 시간대 없는 KST 기준 값으로 둔다.

모델 자체 검증
- slack 메시지 3~10개, 시간순, created_at = 첫 메시지 시각
- ticket resolved_at·댓글 시각은 created_at 이후
- minutes 참석자 2명 이상, 중복 없음
"""

import re
from collections.abc import Iterator
from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.schema.knowledge import NonBlankStr, SourceType

SLACK_MESSAGES_MIN = 3
SLACK_MESSAGES_MAX = 10

DOC_ID_PATTERNS: dict[SourceType, str] = {
    SourceType.SLACK: r"^SL-\d{4}$",
    SourceType.TICKET: r"^TK-\d{4}$",
    SourceType.GUIDE: r"^GD-\d{3}$",
    SourceType.MINUTES: r"^MN-\d{3}$",
}
DOC_ID_PREFIX: dict[SourceType, tuple[str, int]] = {
    SourceType.SLACK: ("SL", 4),
    SourceType.TICKET: ("TK", 4),
    SourceType.GUIDE: ("GD", 3),
    SourceType.MINUTES: ("MN", 3),
}

TicketCategory = Literal["장애", "계정", "권한신청", "문의"]
TicketStatus = Literal["RESOLVED", "REJECTED"]


def format_doc_id(source_type: SourceType, number: int) -> str:
    prefix, width = DOC_ID_PREFIX[source_type]
    return f"{prefix}-{number:0{width}d}"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 구성 요소 (LLM 구조화 출력에도 그대로 쓴다)
# ---------------------------------------------------------------------------


class SlackMessage(_Model):
    slack_user_id: NonBlankStr
    ts: datetime
    text: NonBlankStr


class TicketComment(_Model):
    user_id: NonBlankStr
    created_at: datetime
    text: NonBlankStr


class ActionItem(_Model):
    owner_user_id: NonBlankStr
    task: NonBlankStr
    due_date: date | None


# ---------------------------------------------------------------------------
# 문서
# ---------------------------------------------------------------------------


class _DocumentBase(_Model):
    doc_id: str
    source_type: SourceType
    created_at: datetime
    title: NonBlankStr
    gold_ids: list[str] = Field(min_length=1, description="평가용 숨김 라벨")
    author_user_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        if not re.match(DOC_ID_PATTERNS[self.source_type], self.doc_id):
            raise ValueError(f"doc_id {self.doc_id!r} 형식이 {self.source_type}와 맞지 않습니다")
        if len(set(self.gold_ids)) != len(self.gold_ids):
            raise ValueError("gold_ids에 중복 값이 있습니다")
        return self

    def text_fields(self) -> Iterator[str]:
        """원문 본문 텍스트 (evidence·메타 노출 검사 대상). 필드 단위로 돌려준다."""
        yield self.title

    def dates(self) -> Iterator[date]:
        """원문에 기록된 날짜 (날짜 규칙 검사 대상)."""
        yield self.created_at.date()


class SlackThread(_DocumentBase):
    source_type: Literal[SourceType.SLACK] = SourceType.SLACK
    channel: NonBlankStr
    messages: list[SlackMessage]

    @model_validator(mode="after")
    def _check_messages(self) -> Self:
        n = len(self.messages)
        if not SLACK_MESSAGES_MIN <= n <= SLACK_MESSAGES_MAX:
            raise ValueError(f"메시지 {n}개 (허용 {SLACK_MESSAGES_MIN}~{SLACK_MESSAGES_MAX}개)")
        ts = [m.ts for m in self.messages]
        if ts != sorted(ts):
            raise ValueError("메시지가 시간순이 아닙니다")
        if self.created_at != ts[0]:
            raise ValueError("created_at은 첫 메시지 시각이어야 합니다")
        return self

    def text_fields(self) -> Iterator[str]:
        yield from super().text_fields()
        yield from (m.text for m in self.messages)

    def dates(self) -> Iterator[date]:
        yield from super().dates()
        yield from (m.ts.date() for m in self.messages)


class Ticket(_DocumentBase):
    source_type: Literal[SourceType.TICKET] = SourceType.TICKET
    ticket_no: NonBlankStr
    category: TicketCategory
    status: TicketStatus
    requester_user_id: NonBlankStr
    assignee_user_id: NonBlankStr
    resolved_at: datetime
    description: NonBlankStr
    comments: list[TicketComment]
    resolution: NonBlankStr

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if self.resolved_at < self.created_at:
            raise ValueError("resolved_at이 created_at보다 빠릅니다")
        for i, c in enumerate(self.comments):
            if c.created_at < self.created_at:
                raise ValueError(f"comments[{i}].created_at이 티켓 생성 시각보다 빠릅니다")
        return self

    def text_fields(self) -> Iterator[str]:
        yield from super().text_fields()
        yield self.description
        yield from (c.text for c in self.comments)
        yield self.resolution

    def dates(self) -> Iterator[date]:
        yield from super().dates()
        yield self.resolved_at.date()
        yield from (c.created_at.date() for c in self.comments)


class Guide(_DocumentBase):
    source_type: Literal[SourceType.GUIDE] = SourceType.GUIDE
    version: NonBlankStr
    updated_at: date
    body: NonBlankStr = Field(description="markdown, ## 섹션으로 구분")

    def text_fields(self) -> Iterator[str]:
        yield from super().text_fields()
        yield self.body

    def dates(self) -> Iterator[date]:
        yield from super().dates()
        yield self.updated_at


class Minutes(_DocumentBase):
    source_type: Literal[SourceType.MINUTES] = SourceType.MINUTES
    meeting_date: date
    attendees: list[str] = Field(description="user_id 목록")
    agenda: list[NonBlankStr]
    discussion: NonBlankStr
    decisions: list[NonBlankStr]
    action_items: list[ActionItem]

    @model_validator(mode="after")
    def _check_attendees(self) -> Self:
        if len(self.attendees) < 2:
            raise ValueError("참석자는 2명 이상이어야 합니다")
        if len(set(self.attendees)) != len(self.attendees):
            raise ValueError("attendees에 중복 값이 있습니다")
        return self

    def text_fields(self) -> Iterator[str]:
        yield from super().text_fields()
        yield from self.agenda
        yield self.discussion
        yield from self.decisions
        yield from (a.task for a in self.action_items)

    def dates(self) -> Iterator[date]:
        yield from super().dates()
        yield self.meeting_date


Document = Annotated[SlackThread | Ticket | Guide | Minutes, Field(discriminator="source_type")]

DOCUMENT_MODELS: dict[SourceType, type[_DocumentBase]] = {
    SourceType.SLACK: SlackThread,
    SourceType.TICKET: Ticket,
    SourceType.GUIDE: Guide,
    SourceType.MINUTES: Minutes,
}
