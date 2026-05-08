from bs4 import BeautifulSoup
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def fetch_website_contents(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"Could not fetch {url}"

        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.title.string if soup.title else "No title found"

        text = soup.get_text(separator=" ", strip=True)

        return f"""
URL: {url}

TITLE:
{title}

CONTENT:
{text[:2000]}
"""

    except Exception as e:
        print(f"Failed to fetch content from {url}")
        print(e)

        return ""


def fetch_website_links(url):
    """
    Return the links on the website at the given url
    """

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        links = [link.get("href") for link in soup.find_all("a")]

        return [link for link in links if link]

    except Exception as e:
        print(f"Failed to fetch links from {url}")
        print(e)

        return []

def fetch_page_and_all_relevant_links(url):
    result = fetch_website_contents(url)

    links = fetch_website_links(url)

    for link in links[:5]:
        if link.startswith("/"):
            link = url.rstrip("/") + link

        if link.startswith("http"):
            result += "\n\n"
            result += fetch_website_contents(link)

    return result