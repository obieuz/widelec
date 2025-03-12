# Praktyczne projekty z wykorzystaniem środowisk wirtualnych

## Projekt 1: Prosty scraper wiadomości z venv

### Cel projektu
Stworzymy prostą aplikację, która będzie pobierać najnowsze nagłówki z wybranej strony internetowej.

### Krok 1: Tworzenie środowiska wirtualnego
```bash
#Pójdź do folderu news_scraper
cd news_scraper

# Stwórz środowisko wirtualne
python -m venv .venv

# Aktywuj środowisko
# Windows:
.venv\Scripts\activate
# MacOS/Linux:
source .venv/bin/activate
```

### Krok 2: Instalacja zależności
```bash
pip install requests beautifulsoup4
pip freeze > requirements.txt
```

### Krok 3: Kod aplikacji
Utwórz plik `scraper.py` z następującą zawartością:

```python
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
```

### Krok 4: Uruchomienie projektu
```bash
python scraper.py
```

### Co Ci to daje?
- Praktyczne doświadczenie z tworzeniem i aktywacją środowiska venv
- Zarządzanie zależnościami za pomocą pip
- Tworzenie pliku requirements.txt do replikacji środowiska

---

## Projekt 2: Generator danych testowych z Poetry

### Cel projektu
Stworzymy bibliotekę do generowania realistycznych danych testowych dla aplikacji.

### Krok 1: Instalacja Poetry (jeśli jeszcze nie zainstalowane)
```bash
# Dla systemów Unix/MacOS
curl -sSL https://install.python-poetry.org | python3 -

# Dla Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Krok 2: Inicjalizacja projektu
```bash
# Utwórz nowy projekt
cd data_generator
poetry init
```

### Krok 3: Dodanie zależności
```bash
poetry add faker click
poetry add --dev pytest
```

### Krok 4: Implementacja generatora danych
Edytuj plik `data_generator/data_generator.py`:

```python
import random
from faker import Faker

fake = Faker()

class DataGenerator:
    def generate_person(self):
        """Generuje dane osobowe"""
        return {
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address(),
            "phone": fake.phone_number(),
            "job": fake.job()
        }
    
    def generate_company(self):
        """Generuje dane firmowe"""
        return {
            "name": fake.company(),
            "catchphrase": fake.catch_phrase(),
            "business": fake.bs(),
            "address": fake.address(),
            "website": fake.domain_name()
        }
    
    def generate_dataset(self, type_name, count=10):
        """Generuje zestaw danych"""
        if type_name == "person":
            return [self.generate_person() for _ in range(count)]
        elif type_name == "company":
            return [self.generate_company() for _ in range(count)]
        else:
            raise ValueError(f"Unknown data type: {type_name}")
```

### Krok 5: Dodanie interfejsu wiersza poleceń
Utwórz plik `data_generator/cli.py`:

```python
import json
import click
from .data_generator import DataGenerator

@click.command()
@click.argument('type_name', type=click.Choice(['person', 'company']))
@click.option('--count', '-c', default=10, help='Liczba rekordów do wygenerowania')
@click.option('--output', '-o', default='data.json', help='Nazwa pliku wyjściowego')
def generate(type_name, count, output):
    """Generuje dane testowe i zapisuje je do pliku JSON."""
    generator = DataGenerator()
    data = generator.generate_dataset(type_name, count)
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    click.echo(f"Wygenerowano {count} rekordów typu {type_name} do pliku {output}")

if __name__ == '__main__':
    generate()
```

### Krok 6: Aktualizacja pliku pyproject.toml
Dodaj sekcję dla CLI:

```toml
[tool.poetry.scripts]
generate-data = "data_generator.cli:generate"
```

### Krok 7: Testy
Utwórz plik `tests/test_data_generator.py`:

```python
from data_generator.data_generator import DataGenerator

def test_generate_person():
    generator = DataGenerator()
    person = generator.generate_person()
    assert isinstance(person, dict)
    assert "name" in person
    assert "email" in person

def test_generate_company():
    generator = DataGenerator()
    company = generator.generate_company()
    assert isinstance(company, dict)
    assert "name" in company
    assert "website" in company

def test_generate_dataset():
    generator = DataGenerator()
    dataset = generator.generate_dataset("person", 5)
    assert isinstance(dataset, list)
    assert len(dataset) == 5
```

### Krok 8: Uruchomienie projektu
```bash
# Instalacja projektu w środowisku deweloperskim
poetry install

# Uruchomienie testów
poetry run pytest

# Generowanie danych
poetry run generate-data person --count 20 --output osoby.json
```

### Co Ci to daje?
- Praktyczne doświadczenie z Poetry
- Tworzenie pakietowalnych aplikacji
- Zarządzanie zależnościami i środowiskiem deweloperskim
- Tworzenie interfejsu wiersza poleceń
- Testy jednostkowe

---

## Projekt 3: Analiza danych z Condą

### Cel projektu
Stworzymy środowisko analizy danych do analizy danych pogodowych.

### Krok 1: Instalacja Condy (jeśli jeszcze nie zainstalowana)
Pobierz Minicondę ze strony: https://docs.conda.io/en/latest/miniconda.html

### Krok 2: Tworzenie środowiska
```bash
# Stwórz folder projektu
cd weather_analysis

# Stwórz środowisko Conda
conda create --name weather_env python=3.10

# Aktywuj środowisko
conda activate weather_env
```

### Krok 3: Instalacja pakietów
```bash
conda install pandas matplotlib seaborn notebook

# Można też użyć kanału conda-forge dla niektórych pakietów
conda install -c conda-forge folium

# Biblioteka do pobierania danych pogodowych
pip install meteostat
```

### Krok 4: Zapisanie specyfikacji środowiska
```bash
conda env export > environment.yml
```

### Krok 5: Analiza danych w Jupyter Notebook
Utwórz plik `weather_analysis.ipynb`:

```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analiza danych pogodowych\n",
    "\n",
    "W tym notebooku analizujemy dane pogodowe z biblioteki Meteostat."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Importy\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from datetime import datetime\n",
    "from meteostat import Point, Daily\n",
    "import folium\n",
    "\n",
    "# Konfiguracja wykresu\n",
    "plt.style.use('ggplot')\n",
    "sns.set(style=\"darkgrid\")\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pobranie danych pogodowych dla Warszawy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Ustawienie punktu (Warszawa)\n",
    "warsaw = Point(52.2297, 21.0122)\n",
    "\n",
    "# Ustawienie zakresu czasu (ostatni rok)\n",
    "start = datetime(2024, 1, 1)\n",
    "end = datetime(2024, 12, 31)\n",
    "\n",
    "# Pobranie danych\n",
    "data = Daily(warsaw, start, end)\n",
    "data = data.fetch()\n",
    "\n",
    "# Wyświetlenie pierwszych wierszy\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analiza temperatur"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Wykres temperatury\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.plot(data.index, data['tavg'], label='Średnia temperatura', color='blue')\n",
    "plt.plot(data.index, data['tmin'], label='Min. temperatura', color='green')\n",
    "plt.plot(data.index, data['tmax'], label='Max. temperatura', color='red')\n",
    "plt.title('Temperatury w Warszawie')\n",
    "plt.xlabel('Data')\n",
    "plt.ylabel('Temperatura (°C)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analiza opadów"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Wykres opadów\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.bar(data.index, data['prcp'], color='skyblue')\n",
    "plt.title('Opady w Warszawie')\n",
    "plt.xlabel('Data')\n",
    "plt.ylabel('Opady (mm)')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Mapa z lokalizacją"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Tworzenie mapy\n",
    "m = folium.Map(location=[52.2297, 21.0122], zoom_start=10)\n",
    "folium.Marker(\n",
    "    location=[52.2297, 21.0122],\n",
    "    popup='Warszawa',\n",
    "    icon=folium.Icon(color='blue')\n",
    ").add_to(m)\n",
    "m"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

### Krok 6: Uruchomienie notatnika
```bash
jupyter notebook
```

### Co Ci to daje?
- Praktyczne doświadczenie z Condą
- Praca z zaawansowanymi bibliotekami naukowymi
- Tworzenie i eksportowanie środowisk Conda
- Analiza danych w Jupyter Notebook

---

## Projekt 4: Web API z izolowanymi środowiskami (Poetry + Docker) (Bez dockera jak będą problemy na szkolnych)

### Cel projektu
Stworzymy proste API RESTowe, które będzie uruchamiane w kontenerze Docker.

### Krok 1: Inicjalizacja projektu z Poetry
```bash
cd weather_api
poetry init

# Dodanie zależności
poetry add fastapi uvicorn meteostat
```

### Krok 2: Implementacja API
Utwórz plik `main.py`:

```python
from fastapi import FastAPI, HTTPException
from meteostat import Point, Daily
from datetime import datetime, timedelta
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Weather API")

class Location(BaseModel):
    lat: float
    lon: float
    days: int = 7

@app.get("/")
def read_root():
    return {"message": "Weather API jest aktywne"}

@app.post("/weather/")
def get_weather(location: Location):
    try:
        # Ustawienie punktu
        point = Point(location.lat, location.lon)
        
        # Ustawienie zakresu czasu
        end = datetime.now()
        start = end - timedelta(days=location.days)
        
        # Pobranie danych
        data = Daily(point, start, end)
        data = data.fetch()
        
        # Konwersja do formatu JSON
        result = []
        for date, row in data.iterrows():
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "tavg": row.get('tavg'),
                "tmin": row.get('tmin'),
                "tmax": row.get('tmax'),
                "prcp": row.get('prcp'),
                "snow": row.get('snow'),
                "wspd": row.get('wspd')
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Krok 3: Tworzenie pliku Dockerfile
Utwórz plik `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalacja Poetry
RUN pip install poetry

# Kopiowanie plików projektu
COPY pyproject.toml poetry.lock* /app/

# Konfiguracja Poetry do instalacji w systemie
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Kopiowanie kodu aplikacji
COPY . /app/

# Uruchomienie aplikacji
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Krok 4: Tworzenie pliku docker-compose.yml
Utwórz plik `docker-compose.yml`:

```yaml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
```

### Krok 5: Uruchomienie aplikacji
```bash
# Budowanie i uruchomienie kontenera
docker-compose up --build
```

### Krok 6: Testowanie API
Otwórz przeglądarkę i przejdź do `http://localhost:8000/docs` aby zobaczyć interaktywną dokumentację API.

### Co Ci to daje?
- Integracja Poetry z Dockerem
- Tworzenie izolowanych środowisk w kontenerach
- Wdrażanie aplikacji Pythona w kontenerach
- Doświadczenie z FastAPI

---

## Projekt 5: Wielośrodowiskowa aplikacja ML (Conda)

### Cel projektu
Stworzymy środowisko do trenowania i wdrażania modelu uczenia maszynowego.

### Krok 1: Stworzenie środowiska do trenowania modelu
```bash
# Otwórz folder projektu
cd ml_project

# Stwórz środowisko treningowe
conda create --name ml_train python=3.10
conda activate ml_train

# Instalacja pakietów treningowych
conda install pandas scikit-learn matplotlib jupyter
conda install -c conda-forge xgboost
pip install category_encoders

# Zapisz środowisko
conda env export > train_environment.yml
```

### Krok 2: Stworzenie środowiska do wdrożenia modelu
```bash
# Stwórz środowisko wdrożeniowe (lżejsze)
conda create --name ml_deploy python=3.10
conda activate ml_deploy

# Instalacja tylko niezbędnych pakietów
conda install pandas scikit-learn
conda install -c conda-forge flask xgboost

# Zapisz środowisko
conda env export > deploy_environment.yml
```

### Krok 3: Trenowanie modelu
Utwórz plik `train_model.py`:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Wygenerowanie przykładowych danych mieszkaniowych
np.random.seed(42)
n_samples = 1000

# Cechy
districts = ['Śródmieście', 'Mokotów', 'Wola', 'Ursynów', 'Praga']
sizes = np.random.normal(60, 20, n_samples)  # Powierzchnia w m²
rooms = np.random.randint(1, 6, n_samples)  # Liczba pokoi
floors = np.random.randint(0, 11, n_samples)  # Piętro
years = np.random.randint(1950, 2023, n_samples)  # Rok budowy
districts_random = np.random.choice(districts, n_samples)  # Dzielnica

# Cena = funkcja powierzchni, liczby pokoi, piętra i dzielnicy z losowym szumem
base_price = 8000  # Cena bazowa za m²
district_factors = {'Śródmieście': 1.3, 'Mokotów': 1.2, 'Wola': 1.0, 'Ursynów': 1.1, 'Praga': 0.9}
prices = []

for i in range(n_samples):
    district_factor = district_factors[districts_random[i]]
    size_factor = sizes[i]
    room_factor = 1 + (rooms[i] - 2) * 0.05  # Więcej pokoi = lekko wyższa cena
    floor_factor = 1 + (floors[i] / 10) * 0.1  # Wyższe piętro = lekko wyższa cena
    year_factor = 1 + (years[i] - 1950) / (2023 - 1950) * 0.3  # Nowszy budynek = wyższa cena
    
    price = base_price * size_factor * district_factor * room_factor * floor_factor * year_factor
    noise = np.random.normal(0, price * 0.1)  # 10% szumu
    prices.append(price + noise)

# Tworzenie dataframe
df = pd.DataFrame({
    'district': districts_random,
    'size': sizes,
    'rooms': rooms,
    'floor': floors,
    'year': years,
    'price': prices
})

# Zapisanie danych do CSV
df.to_csv('apartments.csv', index=False)
print("Zapisano dane do apartments.csv")

# Przygotowanie danych do modelowania
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie pipeline z enkoderem kategorii, skalowaniem i modelem
pipeline = Pipeline([
    ('encoder', OneHotEncoder(cols=['district'])),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Trenowanie modelu
pipeline.fit(X_train, y_train)

# Ocena modelu
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Rzeczywista cena')
plt.ylabel('Przewidywana cena')
plt.title('Porównanie cen rzeczywistych i przewidywanych')
plt.savefig('model_performance.png')
print("Zapisano wykres do model_performance.png")

# Zapisanie modelu
joblib.dump(pipeline, 'apartment_price_model.pkl')
print("Zapisano model do apartment_price_model.pkl")
```

### Krok 4: Wdrożenie modelu jako API
Utwórz plik `app.py`:

```python
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Sprawdź, czy model istnieje
model_path = 'apartment_price_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model nie został znaleziony w {model_path}. Najpierw uruchom train_model.py.")

# Wczytaj model
model = joblib.load(model_path)

@app.route('/')
def home():
    return """
    <h1>API predykcji cen mieszkań</h1>
    <p>Używaj endpointu /predict z metodą POST, aby przewidzieć cenę mieszkania.</p>
    <p>Przykładowe dane wejściowe:</p>
    <pre>
    {
        "district": "Mokotów",
        "size": 65.5,
        "rooms": 3,
        "floor": 4,
        "year": 2010
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z żądania
        data = request.get_json()
        
        # Walidacja danych
        required_fields = ['district', 'size', 'rooms', 'floor', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Brakujące pole: {field}"}), 400
        
        # Konwersja na DataFrame dla predykcji
        input_data = pd.DataFrame([data])
        
        # Wykonaj predykcję
        prediction = model.predict(input_data)[0]
        
        # Zwróć wynik
        return jsonify({
            "predicted_price": round(prediction, 2),
            "input": data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Krok 5: Instrukcje uruchomienia
Utwórz plik `README.md`:

```markdown
# Projekt predykcji cen mieszkań

## Wymagania
- Conda lub Miniconda

## Konfiguracja środowisk

### Środowisko treningowe
```bash
# Utworzenie środowiska z pliku
conda env create -f train_environment.yml

# Aktywacja środowiska
conda activate ml_train

# Trenowanie modelu
python train_model.py
```

### Środowisko wdrożeniowe
```bash
# Utworzenie środowiska z pliku
conda env create -f deploy_environment.yml

# Aktywacja środowiska
conda activate ml_deploy

# Uruchomienie API
python app.py
```

## Testowanie API
Po uruchomieniu API, możesz przetestować je używając CURL:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"district": "Mokotów", "size": 65.5, "rooms": 3, "floor": 4, "year": 2010}'
```

Lub otwórz przeglądarkę i przejdź do `http://localhost:5000/`
```

### Co Ci to daje?
- Praktyczne doświadczenie z tworzeniem oddzielnych środowisk dla różnych etapów projektu
- Trenowanie modelu ML w jednym środowisku
- Wdrażanie modelu w lżejszym środowisku
- Eksportowanie i odtwarzanie środowisk Conda
