from typing import Any

from llama_index.core.workflow import Event
from pydantic import BaseModel


class QueryReceivedEvent(Event):
    query: str


class Subquery(BaseModel):
    company: str
    query: str


class SubqueriesGeneratedEvent(Event):
    subqueries: list[Subquery]


class QuerySplitErrorEvent(Event):
    error: str
    invalid_output: str
    main_query: str


class QuerySplitOutputEvent(Event):
    output: str
    original_query: str


class ChooseAgentEvent(Event):
    query: str


class RetrieverEvent(Event):
    agent: Any
    query: str


class RetrieverResponseEvent(Event):
    response: str


class RetrieverStartEvent(Event):
    pass


class MockEvent(Event):
    pass


class ProcessedQueryEvent(Event):
    query: str


class IndustryMap(BaseModel):
    company: str
    industry: list[str]
    alias: list[str]
    # notes: list[str]
