from pydantic import BaseModel
from typing import Literal, Optional


# query endpoint

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


# papers endpoint

class PaperMeta(BaseModel):
    paper_id: str
    paper_title: str
    publication_year: Optional[int] = None
    chunk_count: int


# synthesize endpoint 

class SynthesizeRequest(BaseModel):
    claim: str
    paper_ids: Optional[list[str]] = None

class PaperAnalysis(BaseModel):
    paper_id: str
    paper_title: str
    publication_year: Optional[int] = None
    stance: Literal["supports", "contradicts", "mixed", "not_addressed"]
    method_summary: str
    key_finding: str
    reasoning: str
    page_refs: list[str] = []  # e.g. ["p. 4", "p. 7-8"]

class SynthesizeResponse(BaseModel):
    claim: str
    synthesis: str
    stance_counts: dict[str, int]
    per_paper: list[PaperAnalysis]
    chronological: list[PaperAnalysis]  # same data, sorted by year
    ambiguity_note: str = ""