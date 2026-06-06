import os
from bs4 import BeautifulSoup

from src.loader import pdf_loader
from src.parser import parse_xml, chunk


def ingest_pdf(file_path, store):

    xml = pdf_loader(file_path)

    soup = BeautifulSoup(xml, "lxml-xml")

    title = soup.find("title").text

    sections = parse_xml(soup)

    chunks = chunk(sections, title)

    store.add(chunks)

    print(f"Indexed: {title}")