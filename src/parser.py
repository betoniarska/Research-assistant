from bs4 import BeautifulSoup

# parse xml and extract sections
def parse_xml(soup):

    sections = []

    for div in soup.find_all("div"):
        head = div.find("head")
        if head:
            section_title = head.text.strip()
            section_text = div.text.strip()
            sections.append({
                "title": section_title,
                "text": section_text
            })

    return sections


# chunk sections into smaller pieces for better processing by language models
def chunk_text(text, size=800):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# chunk sections and add paper title for context
def chunk(sections, title):

    chunks = []

    # added a unique ID to each chunk for better traceability and to avoid duplicates in the index
    chunk_id = 0

    for sec in sections:
        for chunk in chunk_text(sec["text"]):
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "section": sec["title"],
                "paper_title": title
            })
            chunk_id += 1

    return chunks



