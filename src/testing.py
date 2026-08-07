# quick debug script, run locally
from bs4 import BeautifulSoup

from src.loader import pdf_loader
xml = pdf_loader("data/attention_is.pdf")  # your loader
soup = BeautifulSoup(xml, "lxml-xml")
print(soup.find("teiHeader").prettify()[:3000])