from bs4 import BeautifulSoup

import uuid
import re


SKIP_SECTIONS = {"references", "acknowledgements", "appendix"}

# heuristics to extract publication year and title from GROBID TEI XML, which can be inconsistent depending on PDF metadata quality.
def extract_publication_year(soup):
    """
    Scope strictly to the paper's own publication date in titleStmt/sourceDesc,
    not to any date appearing elsewhere in the header (e.g. citation dates,
    processing timestamps).
    """
    header = soup.find("teiHeader")
    if not header:
        return None

    file_desc = header.find("fileDesc")
    if file_desc:
        date_tag = file_desc.find("date", attrs={"type": "published"})
        if date_tag and date_tag.get("when"):
            match = re.match(r"(\d{4})", date_tag["when"])
            if match:
                return int(match.group(1))

    # narrower fallback — still scoped to fileDesc, not the whole header
    if file_desc:
        date_tag = file_desc.find("date", attrs={"when": True})
        if date_tag:
            match = re.match(r"(\d{4})", date_tag["when"])
            if match:
                return int(match.group(1))

    return None  # don't guess from unscoped header text — better to show "unknown"

# GROBID's <title> can be unreliable, so use heuristics to find the most likely candidate.
def extract_title(soup):
    header = soup.find("teiHeader")
    title = None
    if header:
        title_stmt = header.find("titleStmt")
        if title_stmt:
            title_tag = title_stmt.find("title")
            if title_tag and title_tag.text.strip():
                title = title_tag.text.strip()

    if not title:
        candidates = [t.text.strip() for t in soup.find_all("title") if t.text.strip()]
        title = next((c for c in candidates if len(c) < 120), candidates[0] if candidates else "Unknown title")

    # strip common license/permission boilerplate that GROBID sometimes
    # merges into the title field from the PDF's first-page header/footer
    title = re.sub(r"^.*?permission to reproduce.*?works\.\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^.*?all rights reserved\.\s*", "", title, flags=re.IGNORECASE)

    return title.strip()

# main parser logic
def parse_xml(soup):

    sections = []

    for div in soup.find_all("div"):
        head = div.find("head")
        if not head:
            continue

        title = head.text.strip()
        if any(s in title.lower() for s in SKIP_SECTIONS):
            continue

        paragraphs = []
        for p in div.find_all("p"):
            text = p.get_text(" ", strip=True)
            if not text:
                continue

            # find nearest preceding page break to attribute a page number
            pb = p.find_previous("pb")
            page_num = pb.get("n") if pb and pb.get("n") else None

            paragraphs.append({"text": text, "page": page_num})

        full_text = " ".join(p["text"] for p in paragraphs)
        if len(full_text.split()) < 50:  # skip stubs like figure captions
            continue

        # page range for the section (first and last page seen)
        pages = [p["page"] for p in paragraphs if p["page"]]
        page_range = (pages[0], pages[-1]) if pages else (None, None)

        sections.append({
            "title": title,
            "text": full_text,
            "page_start": page_range[0],
            "page_end": page_range[1],
        })

    return sections


def chunk_text(text, size=400, overlap=100):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap

    return chunks


def chunk(sections, title, paper_id, publication_year=None):

    chunks = []

    for sec in sections:
        for chunk_str in chunk_text(sec["text"]):
            chunks.append({
                "id": str(uuid.uuid4()),
                "paper_id": paper_id,
                "paper_title": title,
                "version_date": publication_year,  # None if unknown — never guessed
                "section": sec["title"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "text": f"{sec['title']}. {chunk_str}",
            })

    print("Sections:", len(sections))
    print("Chunks:", len(chunks))

    return chunks