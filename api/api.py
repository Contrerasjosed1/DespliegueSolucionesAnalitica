from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Inicializar la app Flask
app = Flask(__name__)
CORS(app)  # Permitir solicitudes desde otros orígenes (opcional)

# Cargar el modelo
model = joblib.load('modelo_prediccion_pipeline.pkl')

# Ruta principal para la predicción
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON del cliente
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No se proporcionaron datos"}), 400

        # Convertir los datos en un DataFrame
        input_df = pd.DataFrame([input_data])

        # Realizar la predicción
        prediction = model.predict(input_df)

        # Retornar el resultado
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para verificar que el servidor funciona
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API activa y funcionando"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)