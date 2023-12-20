import os
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


def df_processing(df, faculty_of):
    df = pd.DataFrame(df)
    df.reset_index(drop=True, inplace=True)
    df["faculty_of"] = faculty_of
    if os.path.exists("../data/faculty-raw.csv"):
        raw = pd.read_csv("../data/faculty-raw.csv", sep="|")
        df = pd.concat([df, raw], sort=False)
    df.drop_duplicates(inplace=True)
    df.sort_values(by="name", ascending=True, inplace=True)
    df.to_csv("../data/faculty-raw.csv", index=False, sep="|")


class ChicagoHarris:
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


    def get_data(self):
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
        
        df_processing(df, "HARRIS SCHOOL OF PUBLIC POLICY")


class ColumbiaSIPA:
    def get_directory(self, page):
        url = f"https://www.sipa.columbia.edu/communities-connections/faculty-directory?page={page}"
        html = make_request(url)
        directory = html.find_all("article", {"class": "cc--component-container cc--profile-card"})
        directory = [x.find("a")["href"] for x in directory]
        directory = [f"https://www.sipa.columbia.edu{x}" for x in directory]

        return directory


    def get_name(self, html):
        name = html.find("div", {"class": "f--field f--page-title"}).find("h1").text

        return name


    def get_bio(self, html):
        bio = "".join([x.text for x in html.find("div", {"class": "f--field f--wysiwyg"})]).replace("\n", " ").strip()

        return bio
    

    def get_data(self):
        faculties = self.get_directory("0")
        df = []
        for faculty in faculties:
            f = make_request(faculty)
            name = self.get_name(f)
            bio = self.get_bio(f)
            data = {"name": name, "bio": bio}
            df.append(data)
        
        df_processing(df, "COLUMBIA SCHOOL OF INTERNATIONAL AND PUBLIC AFFAIRS")


class HKS:
    def get_directory(self, page):
        url = f"https://www.hks.harvard.edu/faculty-profiles?page={page}"
        html = make_request(url)
        directory = html.find_all("div", {"class": "views-row"})
        directory = [x.find("a")["href"] for x in directory]
        directory = [f"https://www.hks.harvard.edu{x}" for x in directory]

        return directory
    
    def get_name(self, html):
        name = html.find("h1", {"class": "page-title"}).text
        
        return name
    
    def get_bio(self, html):
        bio = "".join([x.text for x in html.find("div", {"class": "profile-tab tab-selected"}).find_all("p")]).strip()

        return bio


    def get_data(self):
        faculties = self.get_directory("0")
        df = []
        for faculty in faculties:
            f = make_request(faculty)
            name = self.get_name(f)
            bio = self.get_bio(f)
            data = {"name": name, "bio": bio}
            df.append(data)
        
        df_processing(df, "HARVARD KENNEDY SCHOOL")
    

class IndianaOneill:
    def get_directory(self):
        url = "https://oneill.indiana.edu/faculty-research/directory/index.html"
        html = make_request(url)
        directory = html.find("div", {"id": "filter-results"})
        directory = [x["href"] for x in directory.find_all("a")]
        directory = [f"https://oneill.indiana.edu{x}" for x in directory]

        return directory
    

    def get_name(self, html):
        name = html.find("h2", {"class": "title"}).text

        return name


    def get_bio(self, html):
        bio = "".join([x.text for x in html.find_all("li", {"itemprop": "degrees"})]).strip()

        return bio
    

    def get_data(self):
        faculties = self.get_directory()
        df = []
        for faculty in faculties:
            f = make_request(faculty)
            name = self.get_name(f)
            bio = self.get_bio(f)
            data = {"name": name, "bio": bio}
            df.append(data)
        
        df_processing(df, "INDIANA UNIVERSITY O'NEILL SCHOOL")


class MichiganFord:
    def get_directory(self, page):
        url = f"https://fordschool.umich.edu/directory?page={page}"
        html = make_request(url)
        directory = [x.find("a")["href"] for x in html.find_all("div", {"class": "view__row"})]
        directory = [f"https://fordschool.umich.edu{x}" for x in directory]

        return directory
    

    def get_name(self, html):
        name = html.find("strong").text.strip()

        return name


    def get_bio(self, html):
        bio = "".join([x.text for x in html.select("#expertise-panel > div:nth-child(1) > div:nth-child(1)")[0]]).strip()

        return bio


    def get_data(self):
        faculties = self.get_directory("0")
        df = []
        for faculty in faculties:
            f = make_request(faculty)
            name = self.get_name(f)
            bio = self.get_bio(f)
            data = {"name": name, "bio": bio}
            df.append(data)
        
        df_processing(df, "FORD SCHOOL OF PUBLIC POLICY")
