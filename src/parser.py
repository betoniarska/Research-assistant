from bs4 import BeautifulSoup

import uuid



# parse xml and extract sections
SKIP_SECTIONS = {"references", "acknowledgements", "appendix"}

def parse_xml(soup):
    sections = []
    for div in soup.find_all("div"):
        head = div.find("head")
        if not head:
            continue
        title = head.text.strip()
        if any(s in title.lower() for s in SKIP_SECTIONS):
            continue
        paragraphs = [p.get_text(" ", strip=True) for p in div.find_all("p")]
        text = " ".join(paragraphs)
        if len(text.split()) < 50:  # skip stubs like figure captions
            continue
        sections.append({"title": title, "text": text})
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
def chunk(sections, title, paper_id):

    chunks = []

    # added a unique ID to each chunk for better traceability and to avoid duplicates in the index

    for sec in sections:
        for chunk in chunk_text(sec["text"]):
            chunks.append({
                "id": str(uuid.uuid4()), # unique ID for the chunk
                "paper_id": paper_id, # link to the original paper
                "paper_title": title, 
                "section": sec["title"], 
                "text": f"{sec['title']}. {chunk}"
            })

            

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
        #print()
        #print(c["section"])
        #print(c["text"][:200])
        pass

    return chunks



