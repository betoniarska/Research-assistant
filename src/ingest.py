import os
from bs4 import BeautifulSoup

from src.loader import pdf_loader
from src.parser import parse_xml, chunk, extract_publication_year, extract_title

import hashlib


def get_paper_id(file_path):
    return hashlib.md5(file_path.encode()).hexdigest()

# Ingest and index a PDF file
def ingest_pdf(file_path, store):

    xml = pdf_loader(file_path)

    soup = BeautifulSoup(xml, "lxml-xml")

    title = extract_title(soup)
    publication_year = extract_publication_year(soup)

    sections = parse_xml(soup)

    chunks = chunk(
        sections,
        title,
        paper_id=get_paper_id(file_path),
        publication_year=publication_year,
    )

    store.add(chunks)

    year_str = publication_year if publication_year else "unknown year"
    print(f"Indexed: {title} ({year_str})")