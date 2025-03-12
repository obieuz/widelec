# Przewodnik projektów z wirtualnymi środowiskami w Pythonie

Ten przewodnik zawiera instrukcje do pięciu praktycznych projektów Python, które pomogą Ci zrozumieć i efektywnie korzystać z różnych narzędzi do zarządzania środowiskami wirtualnymi. Zamiast konkretnych komend, skupiamy się na krokach koncepcyjnych.

## Projekt 1: Scraper wiadomości (venv)

### Zadania do wykonania

1. Otwórz folder projektu o nazwie `news_scraper`
2. Stwórz i aktywuj środowisko wirtualne `venv` wewnątrz tego folderu
3. Zainstaluj biblioteki `requests` i `beautifulsoup4`
4. Wygeneruj plik `requirements.txt` na podstawie zainstalowanych bibliotek
5. Stwórz plik `scraper.py` z następującą funkcjonalnością:
    - Pobieranie strony internetowej (np. Hacker News)
    - Parsowanie HTML za pomocą BeautifulSoup
    - Wyodrębnianie nagłówków wiadomości
    - Wyświetlanie listy nagłówków

### Efekt końcowy

Po uruchomieniu programu, powinieneś zobaczyć listę najnowszych nagłówków z wybranej strony internetowej.

---

## Projekt 2: Generator danych testowych (Poetry)

### Potrzebne narzędzia
- Poetry

### Zadania do wykonania

1. Otwórz projekt Poetry o nazwie `data_generator`
2. Dodaj zależności: `faker` (do generowania danych) i `click` (dla interfejsu CLI)
3. Dodaj zależność deweloperską: `pytest`
4. Stwórz klasę `DataGenerator` w głównym module, która:
    - Generuje dane osobowe (imię, email, adres, telefon, zawód)
    - Generuje dane firmowe (nazwa firmy, slogan, branża, adres, strona www)
    - Tworzy zestawy danych wybranego typu o określonej liczbie rekordów
5. Stwórz interfejs wiersza poleceń (CLI) używając biblioteki Click
6. Skonfiguruj plik `pyproject.toml` aby zawierał skrypt `generate-data`
7. Napisz testy dla generatora danych

### Efekt końcowy

Narzędzie CLI, które generuje pliki JSON z realistycznymi danymi testowymi dla osób lub firm.

---

## Projekt 3: Analiza danych pogodowych (Conda)

### Potrzebne narzędzia

- Miniconda lub Anaconda
- Jupyter Notebook
- Edytor kodu

### Zadania do wykonania

1. Otwórz folder projektu o nazwie `weather_analysis`
2. Stwórz nowe środowisko Conda z Pythonem 3.10
3. Zainstaluj w środowisku:
    - pandas, matplotlib, seaborn, jupyter notebook
    - folium (z kanału conda-forge)
    - meteostat (poprzez pip)
4. Wyeksportuj specyfikację środowiska do pliku `environment.yml`
5. Otwórz Jupyter Notebook z analizą danych pogodowych, który:
    - Używa biblioteki Meteostat do pobrania danych pogodowych
    - Wizualizuje temperatury (średnie, min, max) na wykresie liniowym
    - Tworzy wykres słupkowy opadów
    - Generuje interaktywną mapę z lokalizacją za pomocą Folium


---

## Projekt 4: API pogodowe (Poetry + Docker)

### Potrzebne narzędzia

- Poetry
- Docker i Docker Compose

### Zadania do wykonania

1. Otwórz folder projektu o nazwie `weather_api`
2. Zainicjuj projekt Poetry i dodaj zależności: FastAPI, Uvicorn, Meteostat
3. Otwórz plik `main.py` implementujący API, które:
    - Udostępnia endpoint do pobierania danych pogodowych
    - Przyjmuje parametry: szerokość i długość geograficzna, liczba dni
    - Zwraca dane w formacie JSON
4. Stwórz plik `Dockerfile` konfigurujący obraz Docker z Twoją aplikacją
5. Stwórz plik `docker-compose.yml` do łatwego uruchamiania aplikacji

### Efekt końcowy

Konteneryzowane API REST, które dostarcza dane pogodowe dla określonej lokalizacji, gotowe do wdrożenia w dowolnym środowisku obsługującym Docker.

---

## Projekt 5: Aplikacja predykcji cen mieszkań (Conda z wieloma środowiskami)

### Potrzebne narzędzia

- Miniconda lub Anaconda

### Zadania do wykonania

1. Stwórz nowy folder projektu o nazwie `ml_project`
2. Stwórz dwa oddzielne środowiska Conda:
    - `ml_train` - do trenowania modelu, z bibliotekami do analizy danych i wizualizacji
    - `ml_deploy` - do wdrożenia modelu, lżejsze, tylko z niezbędnymi bibliotekami
3. Otwórz plik `train_model.py` w którym:
    - Wygenerujesz przykładowe dane mieszkaniowe (dzielnica, powierzchnia, liczba pokoi, piętro, rok budowy, cena)
    - Przygotujesz dane do modelowania
    - Zbudujesz pipeline z enkoderem kategorii, skalowaniem i modelem RandomForest
    - Wytreńujesz model i ocenisz jego jakość
    - Wizualizujesz wyniki
    - Zapiszesz model do pliku
4. Stwórz plik `app.py` implementujący API z Flask, które:
    - Wczytuje zapisany model
    - Udostępnia endpoint `/predict` do predykcji cen
    - Przyjmuje dane o mieszkaniu jako JSON
    - Zwraca przewidywaną cenę
5. Stwórz plik README.md z instrukcjami uruchomienia

### Efekt końcowy

Kompletny projekt uczenia maszynowego z oddzielnymi środowiskami do trenowania i wdrażania modelu oraz API do przewidywania cen mieszkań.
