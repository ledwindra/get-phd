import pandas as pd
import requests
from bs4 import BeautifulSoup

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

def raw_data(name, bio):
    data = {"name": name, "bio": bio}

    return data


class HarrisChicago:
    def get_directory(self, page):
        directory = make_request(f"https://harris.uchicago.edu/directory?field_last__value=All&page={page}")
        directory = directory.find("div", {"id": "block-harris-theme-content"})
        directory = [x["href"] for x in directory.find_all("a")]
        directory = [x for x in directory if "/directory/" in x]
        directory = [f"https://harris.uchicago.edu{x}" for x in directory]

        return directory


    def get_name(self, html):
        name = html.select(".node--hero--title > h1:nth-child(1) > span:nth-child(1)")[0].text.upper()

        return name


    def get_bio(self, html):
        bio = html.find("div", {"class": "node--content--main--biography"})
        bio = bio.find_all("p")
        bio = " ".join([x.text.upper() for x in bio]).strip()
        
        return bio
    
    def harris_data(self, name, bio):
        page = 0
        next = True
        df = []
        while next == True:
            pager = make_request(f"https://harris.uchicago.edu/directory?page={page}")
            next = pager.find("li", {"class": "pager__item pager__item--next"})
            next = (next != None)
            faculties = self.get_directory(page)
            for faculty in faculties:
                try:
                    f = make_request(faculty)
                    name = self.get_name(f)
                    bio = self.get_bio(f)
                    data = {"name": name, "bio": bio}
                    df.append(data)
                except AttributeError:
                    pass
            page += 1
        df = pd.DataFrame(df)
        df.sort_values(by="name", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["faculty_of"] = "HARRIS SCHOOL OF PUBLIC POLICY"
        df.to_csv("faculty-raw.csv", index=False, sep="|")
    
class UCSDGPS:
    pass
