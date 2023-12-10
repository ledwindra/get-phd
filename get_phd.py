import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def make_request(url):
    status_code = None
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0"}
    while status_code != 200:
        try:
            res = requests.get(url, timeout=10, headers=headers)
            html = BeautifulSoup(res.content, "html.parser")
            status_code = 200
            
            return html
        except:
            continue


def get_directory(page):
    directory = make_request(f"https://harris.uchicago.edu/directory?field_last__value=All&page={page}")
    # find a list of faculty members from each page
    directory = directory.find("div", {"id": "block-harris-theme-content"})
    directory = [x["href"] for x in directory.find_all("a")]
    directory = [x for x in directory if "/directory/" in x]
    directory = [f"https://harris.uchicago.edu{x}" for x in directory]

    return directory


def get_name(html):
    name = html.select(".node--hero--title > h1:nth-child(1) > span:nth-child(1)")[0].text.upper()

    return name


def get_bio(html):
    bio = html.find("div", {"class": "node--content--main--biography"})
    bio = bio.find_all("p")
    paragraph = []
    for b in bio: paragraph.append(b.text)
    paragraph = "".join(paragraph).lower()
    
    return paragraph


def get_phd(context, model_name):
    phd = re.findall("phd|ph.d.|doctorate| doctoral", context)
    if len(phd) > 0:
        question = "Where did the person receive a doctorate/PhD degree?"
        QA_input = {"question": question, "context": context}
        nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
        res = nlp(QA_input)["answer"].upper()
    else:
        res = None
    
    return res
    
def create_data(page):
    urls = get_directory(page)
    data = []
    for url in urls:
        html = make_request(url)
        name = get_name(html)
        try:
            context = get_bio(html)
            res = get_phd(context, "deepset/tinyroberta-squad2")
            data.append({"name": name, "phd": res})
        except AttributeError:
            data.append({"name": name, "phd": None})
    
    df = pd.DataFrame(data)
    df = df.sort_values(by="name", ascending=True)

    return df


if __name__ == "__main__":
    page = 0
    df = pd.DataFrame([])
    while page <= 13:
        df = pd.concat([df, create_data(page)], sort=False)
        page += 1
    df.to_csv("data/get-phd.csv", index=False, sep="|")