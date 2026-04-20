import os
from bs4 import BeautifulSoup
import xml
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from src.loader import pdf_loader
from src.parser import parse_xml, chunk
from src.vector_index import create_index, search
from src.prompt import ask_llm



file_path = os.path.join(os.path.dirname(__file__), "../data/attention_is.pdf")
file_path = os.path.abspath(file_path)

# load pdf and convert to xml using grobid
data_xml = pdf_loader(file_path)

# parse xml and extract sections
soup = BeautifulSoup(data_xml, "lxml-xml")

title = soup.find("title").text
abstract = soup.find("abstract").text

sections = parse_xml(soup)
chunks = chunk(sections, title)

#print(chunks[0])

for sec in sections:
    #print(sec["title"])
    pass

# create vector index
index, model = create_index(chunks)

# search for relevant sections using a query
query = "What is the main contribution of this paper?"
results = search(query, model, index, chunks)

print(query, '\n')

for r in results:
    print("\n---")
    print("Section:", r["section"])
    print(r["text"][:600])


answer, sources = ask_llm(query, model, index, chunks)

print(answer)

