# Środowiska wirtualne
Zacznijmy sobie od tego czym są środowiska wirtualne i po co mamy się męczyć w jakieś dziwne venvy, poetry (brzmi to jak jakiś przekształcony byt z polskiego) i jakieś dzikie condy (uwaga gryzie, a potem pożera w całości).

## Po co to komu?
### Posłużmy się historyjką o budowniczych:
#### Oto nasi główni bohaterowie:  
![Pasted image 20250305081122](https://github.com/user-attachments/assets/08ba4867-d401-4f75-9cca-60580ae63211)  
**Bob i Bob Jr (Są bardzo podobni do siebie)**  
![Pasted image 20250305081253](https://github.com/user-attachments/assets/8bc9ebc5-f296-4d39-af9d-d622193f7614)  
**Franklin**
#### Historyjka
Bob Jr. uwielbia swoje zamki w powietrzu. Jego świat to lekkie, zwiewne konstrukcje, które unoszą się wśród chmur i wyglądają jak z bajki. Nie przejmuje się grawitacją ani fundamentami, bo przecież jego zamki unoszą się samoistnie. Z kolei Franklin i jego ekipa budują rzeczy solidne – ich fortece są zbudowane z kamienia, mają grube mury, fosy, a każda cegła jest osadzona na solidnym fundamencie.

Franklin potrzebuje doświadczonego budowniczego, który zna się na rzeczy. Bob (starszy brat Boba Jr.) byłby idealnym kandydatem – jest mistrzem w budowaniu niezdobytych warowni. Ale jest problem: jeśli na budowie pojawi się Bob Jr., to zacznie upierać się przy swoim stylu i próbować stawiać zamki w powietrzu zamiast solidnych fortec. Co więcej, jego metody budowy będą kłócić się z metodami Boba, przez co materiały zaczną znikać, fundamenty się rozmywać, a cały projekt zamieni się w jeden wielki bałagan.

Tu właśnie wkraczają środowiska wirtualne! Dzięki nim można zamknąć Boba Jr. w jego własnym świecie, gdzie będzie budować swoje podniebne zamki, a Franklin może spokojnie zatrudnić starszego Boba i razem stworzyć fortecę nie do zdobycia. Każdy ma swoje miejsce i swoje materiały, bez niepotrzebnych konfliktów.

## Jak je stworzyć?
Skoro już wiemy, po co nam środowiska wirtualne, to jak pozbyć się Boba Jr. z budowy? Przecież nie możemy pozwolić, żeby znowu wszystko popsuł!

### Dlaczego to takie ważne?

Każda aplikacja w Pythonie potrzebuje różnych narzędzi – bibliotek i modułów, które pomagają jej działać. Problem pojawia się, gdy różne aplikacje wymagają innych wersji tych samych bibliotek. Wyobraź sobie, że aplikacja A potrzebuje cegieł starego typu, a aplikacja B chce korzystać z nowoczesnych materiałów budowlanych. Jeśli na placu budowy pojawi się dostawa tylko jednego rodzaju cegieł, to jedna z aplikacji przestanie działać.

No i mamy problem! Na szczęście tutaj z pomocą przychodzą środowiska wirtualne. To jak zamknięcie każdej budowy w oddzielnym magazynie – aplikacja A ma swoje własne cegły, a aplikacja B dostaje dokładnie to, czego potrzebuje. Dzięki temu nie ma kłótni o materiały, a każda budowa może działać niezależnie.

### Tworzenie środowiska wirtualnego

Żeby stworzyć oddzielny plac budowy dla każdej aplikacji, używamy modułu `venv`. On pozwala nam wygenerować oddzielny magazyn, w którym będą tylko te cegły, których aktualnie potrzebujemy.

#### Jak to zrobić?

Wybieramy miejsce, gdzie chcemy postawić nasze nowe środowisko, i uruchamiamy polecenie:

```
python -m venv .venv
```

To stworzy katalog `venv`, który będzie zawierał wszystko, co jest potrzebne do pracy – od narzędzi po odpowiednie wersje materiałów.

Popularnym miejscem na środowiska wirtualne jest katalog `.venv`. Dzięki temu jest on ukryty przed niepotrzebnym bałaganem, ale jednocześnie od razu wiadomo, do czego służy.

### Aktywowanie środowiska

Magazyn zbudowany, ale trzeba go jeszcze otworzyć! W zależności od systemu robimy to trochę inaczej:

- **Windows:**
    
    ```
    .venv\Scripts\activate
    ```
    
- **MacOS / Linux:**
    
    ```
    source .venv/bin/activate
    ```
    

Po aktywacji powłoka zacznie pokazywać nazwę środowiska, w którym jesteśmy, a polecenie `python` będzie korzystać z odpowiedniej wersji.

Dzięki temu nie musimy się martwić, że Bob Jr. przyjdzie i zacznie budować coś na własną rękę! W naszym nowym środowisku mamy dokładnie to, czego potrzebujemy, i nic więcej.

### Kończymy budowę

Jeśli skończyliśmy pracę i chcemy zamknąć nasze środowisko, wystarczy wpisać:

```
deactivate
```

I gotowe! Środowisko czeka na kolejne użycie, a my możemy wrócić do innych projektów bez ryzyka, że coś się pomiesza.

## Inne narzędzia: Poetry i Conda

### Poznajmy nowych bohaterów naszej budowlanej opowieści

#### Poetry - Architekt doskonały

Wyobraź sobie architekta, który nie tylko projektuje, ale też sam organizuje cały plac budowy, rozlicza materiały i pilnuje, żeby każda cegła była we właściwym miejscu. To właśnie Poetry!

Podczas gdy `venv` jest jak prosty magazyn narzędzi, Poetry to prawdziwy kombinat budowlany. Zamiast osobnych list materiałów (`requirements.txt`), Poetry trzyma wszystko w jednym, eleganckim pliku `pyproject.toml`. To jak centralny rejestr wszystkich materiałów i narzędzi potrzebnych do budowy.

##### Jak go użyć?

```bash
# Instalacja Poetry - jak sprowadzenie mistrza budowlanego 
curl -sSL https://install.python-poetry.org | python3 - # Stworzenie nowego projektu 
poetry new moj_zamek 
cd moj_zamek # Dodanie biblioteki jak dodanie nowego narzędzia do zestawu 
poetry add requests
```

Poetry dba o to, żeby Bob Jr. miał dokładnie takie same narzędzia za każdym razem, gdy zechce postawić swój zamek w powietrzu. Żadnych niespodzianek, żadnych różnic między środowiskami!

#### Conda - Wielki Magazynier

A co jeśli nasz Bob Jr. nie chce budować tylko zamków w Pythonie? Co jeśli marzy o projektach w R, C++ lub innych egzotycznych technologiach? Tu z pomocą przychodzi Conda - prawdziwy wielogatunkowy magazynier!

Conda to nie tylko magazyn dla Pythona. To przestronny, wielopoziomowy magazyn, gdzie każdy projekt może dostać dokładnie takie narzędzia, jakich potrzebuje - bez względu na język czy technologię.

##### Jak go użyć?

```bash
# Instalacja Condy - jak otwarcie gigantycznego, wielofunkcyjnego magazynu 
# Pobierz Minicondę ze strony: https://docs.conda.io/en/latest/miniconda.html 
# Stworzenie środowiska jak przygotowanie osobnego piętra magazynu 
conda create --name moj_zamek python=3.10 
conda activate moj_zamek # Dodanie narzędzi jak komplety specjalistycznego sprzętu 
conda install numpy pandas
```

Conda dba o to, żeby Bob Jr. mógł budować swoje zamki nie tylko w Pythonie, ale i w dowolnej technologii, jaką sobie wymarzy!

### Kiedy użyć jakiego narzędzia?

#### venv: Gdy budujesz mały, prosty domek

- Szybkie projekty
- Proste aplikacje
- Gdy nie potrzebujesz skomplikowanej logistyki

#### Poetry: Gdy projektujesz kompleksowy zamek

- Duże projekty
- Potrzebujesz precyzyjnego zarządzania zależnościami
- Chcesz mieć powtarzalne środowisko
- Planujesz publikować własne biblioteki

#### Conda: Gdy budujesz międzynarodowe centrum multitechnologiczne

- Projekty naukowe
- Analiza danych
- Sztuczna inteligencja
- Projekty wymagające narzędzi z różnych języków

### Epilog

Bez względu na to, czy używasz `venv`, `Poetry` czy `Condy`, pamiętaj: każde środowisko wirtualne to bezpieczna przestrzeń dla Twojego kodu. Bob Jr. może eksperymentować, a Franklin może spokojnie budować swoje stabilne fortece!
# Dodatek: Szczegółowy przewodnik instalacji

## Przygotowanie terenu budowy

### Przed rozpoczęciem

Zanim zaczniemy, upewnij się, że masz zainstalowanego Pythona. Sprawdź wersję:

```bash
python --version
```

### Wymagania wstępne

- Python 3.7 lub nowszy
- pip (zazwyczaj instalowany wraz z Pythonem)
- Dostęp do terminala/konsoli

## 1. Poetry - Instalacja krok po kroku (Na windowsie może być problem)

### Instalacja automatyczna

```bash
# Dla systemów Unix/MacOS
curl -sSL https://install.python-poetry.org | python3 -

# Dla Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Weryfikacja instalacji

```bash
poetry --version
```

### Podstawowe komendy Poetry

#### Tworzenie nowego projektu

```bash
# Stworzenie nowego projektu
poetry new nazwe_projektu

# Przejście do katalogu projektu
cd nazwe_projektu
```

#### Zarządzanie zależnościami

```bash
# Dodanie biblioteki
poetry add requests

# Dodanie biblioteki deweloperskiej
poetry add --dev pytest

# Instalacja wszystkich zależności
poetry install

# Usunięcie biblioteki
poetry remove requests
```

#### Uruchamianie środowiska

```bash
# Aktywacja środowiska
poetry shell

# Uruchomienie skryptu w środowisku
poetry run python moj_skrypt.py
```

## 2. Conda - Instalacja krok po kroku

### Wybór dystrybucji

Masz dwie główne opcje:

1. **Miniconda** - lekka wersja, tylko podstawowe narzędzia
2. **Anaconda** - pełen pakiet, wiele preinstalowanych bibliotek naukowych

### Instalacja Miniconda

#### Windows

1. Pobierz instalator ze strony: https://docs.conda.io/en/latest/miniconda.html
2. Uruchom instalator
3. Zaznacz opcję dodania Condy do ścieżki systemowej

#### MacOS/Linux

```bash
# Pobierz skrypt instalacyjny
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Nadaj uprawnienia
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Zainstaluj
./Miniconda3-latest-Linux-x86_64.sh
```

### Weryfikacja instalacji

```bash
# Sprawdzenie wersji
conda --version

# Aktualizacja Condy
conda update conda
```

### Podstawowe komendy Condy

#### Zarządzanie środowiskami

```bash
# Stworzenie środowiska
conda create --name nazwe_projektu python=3.10

# Aktywacja środowiska
conda activate nazwe_projektu

# Dezaktywacja środowiska
conda deactivate

# Lista wszystkich środowisk
conda env list
```

#### Zarządzanie pakietami

```bash
# Instalacja pakietu
conda install numpy

# Instalacja pakietu z konkretnego kanału
conda install -c conda-forge pandas

# Lista zainstalowanych pakietów
conda list

# Eksport środowiska do pliku
conda env export > environment.yml

# Odtworzenie środowiska z pliku
conda env create -f environment.yml
```

## Porównanie narzędzi

|Funkcja|venv|Poetry|Conda|
|---|---|---|---|
|Izolacja środowisk|✓|✓|✓|
|Zarządzanie zależnościami|Ręczne|Automatyczne|Automatyczne|
|Wsparcie dla wielu języków|Nie|Nie|Tak|
|Łatwość użycia|Podstawowa|Zaawansowana|Zaawansowana|
|Idealne dla|Małe projekty|Projekty Python|Nauka, ML, wielojęzyczne|

## Wskazówki dodatkowe

### Wybór narzędzia

- **venv**: Szybkie, proste projekty
- **Poetry**: Złożone projekty Python, zarządzanie pakietami
- **Conda**: Projekty naukowe, ML, wielojęzyczne

### Best practices

- Zawsze używaj środowisk wirtualnych
- Regularnie aktualizuj zależności
- Eksportuj konfigurację środowiska
- Dodawaj `.venv`, `poetry.lock`, `environment.yml` do `.gitignore`

## Troubleshooting

### Najczęstsze problemy

- Konflikty wersji Pythona
- Brak uprawnień
- Problemy z instalacją pakietów

### Rozwiązania

- Sprawdzaj wersje Pythona
- Używaj `sudo` z rozwagą
- Aktualizuj pip, setuptools

Pamiętaj: każde narzędzie ma swoje mocne strony. Wybierz to, które najlepiej pasuje do Twojego projektu!
