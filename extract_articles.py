import requests
from bs4 import BeautifulSoup
import re

NUMBER_OF_PAGES = 20 # Number of pages with articles on vesmir.cz

# Extact articles from vesmir.cz, save them to vesmir_articles.txt
with open('vesmir_articles.txt', 'w', encoding='utf-8') as f:

    for num in range (1, NUMBER_OF_PAGES + 1):
        url = "https://vesmir.cz/cz/on-line-clanky/?page=" + str(num)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        for article in soup.find_all("div", class_="col-sm-9"):
            link = article.find("a")['href']
            print(link)
            article_response = requests.get("https://vesmir.cz" + link)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            for text in article_soup.find_all("div", class_="article"): 
                try:
                    perex = text.find("p", class_="perex").text.strip()
                    if perex:
                        f.write(perex + '\n')
                except:
                    pass

                for p in text.find_all("p", class_=""):
                    if not p.text.strip():
                        break
                    f.write(p.text.strip() + '\n')