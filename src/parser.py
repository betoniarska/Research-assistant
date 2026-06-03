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


# chunk text into smaller pieces for better embedding and retrieval (with overlap to preserve context)
def chunk_text(text, size=400, overlap=100):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):

        end = start + size

        chunks.append(
            " ".join(words[start:end])
        )

        start += size - overlap

    return chunks


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

    # evaluate number of sections and chunks for debugging
    print("Sections:", len(sections))
    print("Chunks:", len(chunks))

    for c in chunks:
        if c["section"] == "Input-Input Layer5":
            print(c)
        else:
            print("no such section: Input-Input Layer5")

    # print first 20 chunks for debugging
    for c in chunks[:20]:
        print()
        print(c["section"])
        print(c["text"][:200])

    return chunks



