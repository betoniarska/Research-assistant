import os
from bs4 import BeautifulSoup

from src.loader import pdf_loader
from src.parser import parse_xml, chunk
from src.vector_store.faiss_store import FAISSStore
from src.prompt import ask_llm


file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/attention_is.pdf")
)



def ingest_pdf(file_path, store):

    xml = pdf_loader(file_path)

    soup = BeautifulSoup(xml, "lxml-xml")

    title = soup.find("title").text

    sections = parse_xml(soup)

    chunks = chunk(sections, title)

    index = store.build(chunks)

    store.save()

    print(f"Indexed paper: {title}")


def query_rag(question, store):

    store.load()

    results = store.search(question, k=5)

    answer, _ = ask_llm(question, results)

    return answer, results



if __name__ == "__main__":

    store = FAISSStore()


    ingest_pdf(file_path, store)

    query = "Simply list the main keywords of the paper in a bullet point list"

    answer, sources = query_rag(query, store)

    print("\nQUESTION:\n", query)
    print("\nANSWER:\n", answer)

    for s in sources:
        print("\n---")
        print(s["section"])
        print(s["text"][:300])