import requests
from bs4 import BeautifulSoup

def get_news_headlines(url):
    # Pobierz stronę
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    # Parsuj HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Znajdź nagłówki (dostosuj selektor do strony)
    headlines = []
    for headline in soup.select('h2.title'):
        headlines.append(headline.text.strip())
    
    return headlines

if __name__ == "__main__":
    url = "https://news.ycombinator.com/"  # Przykład: Hacker News
    headlines = get_news_headlines(url)
    
    print("Dzisiejsze nagłówki:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")