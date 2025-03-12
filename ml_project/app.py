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