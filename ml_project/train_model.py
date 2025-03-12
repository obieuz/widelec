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