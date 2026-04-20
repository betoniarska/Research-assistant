


from bs4 import BeautifulSoup

sections = []

# parse xml and extract sections
def parse_xml(soup):

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

    for sec in sections:
        for chunk in chunk_text(sec["text"]):
            chunks.append({
                "text": chunk,
                "section": sec["title"],
                "paper_title": title
            })

    return chunks



