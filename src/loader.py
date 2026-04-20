import xml
import requests

# pdf loader

def pdf_loader(file_path):
    url = "http://localhost:8070/api/processFulltextDocument"

    with open(file_path, "rb") as f:
        response = requests.post(url, files={"input": f})

    xml = response.text

    return xml

