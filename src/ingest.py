# ingest.py

import os
from bs4 import BeautifulSoup

from src.loader import pdf_loader
from src.parser import parse_xml, chunk

import hashlib

# generate a unique ID for the paper based on the file path (can be used to link all chunks back to the original paper)
def get_paper_id(file_path):
    return hashlib.md5(file_path.encode()).hexdigest()


def ingest_pdf(file_path, store):

    xml = pdf_loader(file_path)

    soup = BeautifulSoup(xml, "lxml-xml")

    title = soup.find("title").text

    sections = parse_xml(soup)

    # chunk sections and add paper title for context
    chunks = chunk(sections, title, paper_id=get_paper_id(file_path))

    store.add(chunks) # does not yet exist in the store, will be added in the next step

    print(f"Indexed: {title}")