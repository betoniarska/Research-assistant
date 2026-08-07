
""" Synthesis module for evaluating claims against research papers collectively.
    Consists of three stages:

    1. Retrieval + reranking of relevant chunks for each paper (CPU-bound)
    2. Structured extraction of stance (extraction prompt), method, finding, reasoning, and page refs (LLM-bound)
    3. Aggregation of stances and synthesis (synthesis prompt) of a concise summary (LLM-bound)
    
"""

import json
import asyncio
from collections import Counter

import numpy as np
from openai import OpenAI

from src.services.embedding_service import embedding_service
from src.services.reranker import Reranker
from src.schemas import PaperAnalysis

client = OpenAI()
reranker = Reranker()



def get_chunks_for_paper(store, paper_id, claim, k=15, top_n=5):
    """
    Retrieve the chunks most relevant to the claim, scoped to a single paper.
    Unlike /query, there's no cross-paper competition for slots here —
    every paper gets its own dedicated retrieval + rerank pass.
    """
    paper_chunks = [c for c in store.chunks if c["paper_id"] == paper_id]
    if not paper_chunks:
        return []

    texts = [c["text"] for c in paper_chunks]

    claim_vec = np.array(embedding_service.encode([claim])).astype("float32")
    chunk_vecs = np.array(embedding_service.encode(texts)).astype("float32")

    # cosine similarity (vectors aren't pre-normalized here, so normalize now)
    claim_norm = claim_vec / np.linalg.norm(claim_vec, axis=1, keepdims=True)
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    sims = (chunk_norm @ claim_norm.T).flatten()

    top_k_idx = sims.argsort()[-k:][::-1]
    candidates = [paper_chunks[i] for i in top_k_idx]

    reranked = reranker.rerank(claim, candidates)
    return reranked[:top_n]




def format_chunks_for_prompt(chunks):
    parts = []
    for c in chunks:
        page_info = ""
        if c.get("page_start"):
            page_info = f" (p. {c['page_start']}" + (
                f"-{c['page_end']})" if c.get("page_end") and c["page_end"] != c["page_start"] else ")"
            )
        parts.append(f"[Section: {c['section']}{page_info}]\n{c['text']}")
    return "\n\n".join(parts)


EXTRACTION_PROMPT = """
You are analyzing a single research paper to evaluate a claim.

Claim being evaluated: "{claim}"

Paper: {paper_title}

Context from this paper:
{context}

Evaluate whether this paper's content supports, contradicts, is mixed on,
or does not address the claim. Base your answer ONLY on the context given.

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "stance": "supports" | "contradicts" | "mixed" | "not_addressed",
  "method_summary": "one sentence describing the paper's relevant method, or empty string if not_addressed",
  "key_finding": "one sentence describing the paper's relevant finding, or empty string if not_addressed",
  "reasoning": "one to two sentences explaining the stance, paraphrased, no verbatim quotes",
  "page_refs": ["p. X", "p. Y-Z"]
}}

If the context does not address the claim at all, use "not_addressed" and
leave method_summary and key_finding as empty strings.
"""



def extract_paper_analysis(claim, paper_id, paper_title, publication_year, chunks):
    """
    Run structured extraction for a single paper against a claim.
    Temperature 0 for determinism — this stage should classify consistently,
    not generate varied prose.
    """
    if not chunks:
        return PaperAnalysis(
            paper_id=paper_id,
            paper_title=paper_title,
            publication_year=publication_year,
            stance="not_addressed",
            method_summary="",
            key_finding="",
            reasoning="No relevant content found for this paper.",
            page_refs=[],
        )

    prompt = EXTRACTION_PROMPT.format(
        claim=claim,
        paper_title=paper_title,
        context=format_chunks_for_prompt(chunks),
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        parsed = {
            "stance": "not_addressed",
            "method_summary": "",
            "key_finding": "",
            "reasoning": "Extraction failed to parse — treated as not addressed.",
            "page_refs": [],
        }

    return PaperAnalysis(
        paper_id=paper_id,
        paper_title=paper_title,
        publication_year=publication_year,
        stance=parsed.get("stance", "not_addressed"),
        method_summary=parsed.get("method_summary", ""),
        key_finding=parsed.get("key_finding", ""),
        reasoning=parsed.get("reasoning", ""),
        page_refs=parsed.get("page_refs", []),
    )



def aggregate(analyses):
    stance_counts = dict(Counter(a.stance for a in analyses))

    # ensure all four stance keys are present even if count is 0 — easier for frontend
    for key in ["supports", "contradicts", "mixed", "not_addressed"]:
        stance_counts.setdefault(key, 0)

    chronological = sorted(
        analyses,
        key=lambda a: (a.publication_year is None, a.publication_year or 0),
    )

    return stance_counts, chronological



SYNTHESIS_PROMPT = """
Claim being evaluated: "{claim}"

Stance counts across {total} papers (these counts are final and exact — do not recompute or estimate them):
- Supports: {supports}
- Contradicts: {contradicts}
- Mixed: {mixed}
- Not addressed: {not_addressed}

Papers with non-supporting stances:
{disputed_list}

Write a concise 3-5 sentence synthesis. State the counts exactly as given above.
If there is disagreement, briefly note which papers diverge and why, based on
the reasoning provided. Do not introduce any numbers not given above.
"""


def write_synthesis(claim, stance_counts, analyses):
    disputed = [a for a in analyses if a.stance in ("contradicts", "mixed")]
    disputed_list = "\n".join(
        f"- {a.paper_title} ({a.publication_year or 'n.d.'}): {a.reasoning}"
        for a in disputed
    ) or "None."

    prompt = SYNTHESIS_PROMPT.format(

        claim=claim,
        total=len(analyses),

        supports=stance_counts["supports"],
        contradicts=stance_counts["contradicts"],
        mixed=stance_counts["mixed"],
        not_addressed=stance_counts["not_addressed"],

        disputed_list=disputed_list,
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content




def get_papers_to_analyze(store, paper_ids=None):
    """
    Return list of (paper_id, paper_title, publication_year) for papers
    to include in the synthesis. None = all indexed papers.
    """
    seen = {}
    for c in store.chunks:
        pid = c["paper_id"]
        if pid not in seen:
            seen[pid] = (pid, c["paper_title"], c.get("publication_year"))

    papers = list(seen.values())

    if paper_ids is not None:
        papers = [p for p in papers if p[0] in paper_ids]

    return papers


async def run_synthesis(claim, store, paper_ids=None):
    papers = get_papers_to_analyze(store, paper_ids)

    if not papers:
        raise ValueError("No matching papers found to analyze.")

    loop = asyncio.get_event_loop()

    async def analyze_one(paper_id, paper_title, publication_year):

        # retrieval + rerank are sync/CPU-bound
        chunks = await loop.run_in_executor(
            None, get_chunks_for_paper, store, paper_id, claim
        )
        # LLM call run in executor also since the OpenAI client here is sync
        return await loop.run_in_executor(
            None, extract_paper_analysis, claim, paper_id, paper_title, publication_year, chunks
        )

    analyses = await asyncio.gather(*[
        analyze_one(pid, title, year) for pid, title, year in papers
    ])

    stance_counts, chronological = aggregate(analyses)
    synthesis = write_synthesis(claim, stance_counts, analyses)

    return synthesis, stance_counts, list(analyses), chronological