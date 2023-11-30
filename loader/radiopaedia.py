import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import DiffbotLoader

# search_query = "atelectasis"
# URL = f"https://radiopaedia.org/search?lang=gb&q={search_query}&scope=articles"
# page = requests.get(URL)
# soup = BeautifulSoup(page.content, "html.parser")
# radio_address = "https://radiopaedia.org"
# all_address = [
#     f"{radio_address}{e['href']}"
#     for e in soup.find_all("a", class_="search-result search-result-article")
# ]
# all_address


# raw_documents = DiffbotLoader(
#     urls=[all_address[0]],
#     api_token=DIFFBOT_API_TOKEN,
# ).load()


class RadiopaediaLoader(DiffbotLoader):
    def __init__(
        self,
        query,
        diffbot_api_token,
    ) -> None:
        URL = f"https://radiopaedia.org/search?lang=gb&q={query}&scope=articles"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        radio_address = "https://radiopaedia.org"
        all_address = [
            f"{radio_address}{e['href']}"
            for e in soup.find_all("a", class_="search-result search-result-article")
        ]
        self.all_address = all_address
        super().__init__(
            urls=all_address,
            api_token=diffbot_api_token,
        )
