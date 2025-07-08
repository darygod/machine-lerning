# app.py - API Flask para predicciones CS:GO
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)  # Permitir requests desde cualquier origen

# Variables globales para el modelo
model_package = None
models = None
scalers = None
features = None
metadata = None

def load_models():
    """Cargar modelos y componentes desde archivos guardados"""
    global model_package, models, scalers, features, metadata
    
    try:
        # Verificar que el archivo existe
        model_path = "saved_models/csgo_model_package.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo de modelo en: {model_path}")
        
        # Cargar el paquete completo
        logger.info("🔄 Cargando modelos...")
        model_package = joblib.load(model_path)
        
        # Extraer componentes
        models = model_package['models']
        scalers = model_package['scalers']
        features = model_package['features']
        metadata = model_package['metadata']
        
        logger.info("✅ Modelos cargados exitosamente")
        logger.info(f"📊 Características disponibles: {len(features['available_features'])}")
        logger.info(f"🎯 Accuracy RF Classifier: {metadata['rf_clf_accuracy']:.4f}")
        logger.info(f"📈 R² RF Regressor: {metadata['rf_reg_r2']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {str(e)}")
        return False

def preprocess_input(data):
    """
    Preprocesar datos de entrada para las predicciones
    
    Args:
        data (dict): Datos del jugador
        
    Returns:
        tuple: (X_processed, error_message)
    """
    try:
        # Convertir a DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Mapear variables categóricas
        categorical_mapping = features['categorical_mapping']
        
        # Codificar Team si está presente
        if 'Team' in df.columns:
            if df['Team'].iloc[0] in categorical_mapping['Team']:
                df['Team_Encoded'] = df['Team'].map(categorical_mapping['Team'])
            else:
                return None, f"Valor de Team inválido. Usar: {list(categorical_mapping['Team'].keys())}"
        else:
            # Valor por defecto si no se proporciona
            df['Team_Encoded'] = 0
        
        # Codificar MatchWinner si está presente
        if 'MatchWinner' in df.columns:
            df['MatchWinner_Encoded'] = df['MatchWinner'].astype(int)
        else:
            # Valor por defecto si no se proporciona
            df['MatchWinner_Encoded'] = 0
        
        # Asegurar que todas las características requeridas estén presentes
        available_features = features['available_features']
        
        # Crear DataFrame con todas las características, rellenando con valores por defecto
        X_processed = pd.DataFrame(columns=available_features)
        
        # Valores por defecto para características faltantes
        default_values = {
            'InternalTeamId': 1,
            'MatchId': 1,
            'RoundId': 1,
            'TravelledDistance': 2000.0,
            'RLethalGrenadesThrown': 0,
            'RNonLethalGrenadesThrown': 0,
            'PrimaryAssaultRifle': 0.8,
            'PrimarySniperRifle': 0.0,
            'PrimaryHeavy': 0.0,
            'PrimarySMG': 0.1,
            'PrimaryPistol': 0.1,
            'FirstKillTime': 30.0,
            'RoundKills': 1,
            'RoundAssists': 0,
            'RoundHeadshots': 0,
            'RoundStartingEquipmentValue': 3000,
            'TeamStartingEquipmentValue': 15000,
            'Team_Encoded': 0,
            'MatchWinner_Encoded': 0
        }
        
        # Llenar características una por una
        for feature in available_features:
            if feature in df.columns:
                X_processed[feature] = df[feature]
            else:
                # Usar valor por defecto
                X_processed[feature] = [default_values.get(feature, 0)]
        
        # Convertir a tipos numéricos
        X_processed = X_processed.astype(float)
        
        return X_processed, None
        
    except Exception as e:
        return None, f"Error en preprocesamiento: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    """Endpoint de inicio con información de la API"""
    if model_package is None:
        return jsonify({
            "error": "Modelos no cargados",
            "message": "Los modelos no están disponibles. Verifica que los archivos existan."
        }), 500
    
    return jsonify({
        "message": "🎮 API de Predicciones CS:GO",
        "version": "1.0.0",
        "status": "✅ Activa",
        "models_loaded": "✅ Sí",
        "metadata": {
            "features_count": len(features['available_features']),
            "rf_classifier_accuracy": f"{metadata['rf_clf_accuracy']:.4f}",
            "rf_regressor_r2": f"{metadata['rf_reg_r2']:.4f}",
            "training_samples_clf": metadata['training_samples_clf'],
            "training_samples_reg": metadata['training_samples_reg']
        },
        "endpoints": {
            "/predict": "Predicción completa (supervivencia + tiempo)",
            "/predict/survival": "Solo predicción de supervivencia",
            "/predict/time": "Solo predicción de tiempo",
            "/health": "Estado de la API",
            "/features": "Lista de características requeridas"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud de la API"""
    if model_package is None:
        return jsonify({"status": "❌ Error", "models_loaded": False}), 500
    
    return jsonify({
        "status": "✅ Saludable",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Obtener lista de características que acepta el modelo"""
    if model_package is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    return jsonify({
        "required_features": features['features_api'],
        "all_features": features['available_features'],
        "categorical_mappings": features['categorical_mapping'],
        "example_input": {
            "Team": "Terrorist",
            "InternalTeamId": 1,
            "MatchId": 100,
            "RoundId": 5,
            "MatchWinner": True,
            "TravelledDistance": 3500.0,
            "RLethalGrenadesThrown": 1,
            "RNonLethalGrenadesThrown": 2,
            "PrimaryAssaultRifle": 0.7,
            "PrimarySniperRifle": 0.1,
            "PrimaryHeavy": 0.0,
            "PrimarySMG": 0.1,
            "PrimaryPistol": 0.1,
            "FirstKillTime": 15.0,
            "RoundKills": 2,
            "RoundAssists": 1,
            "RoundHeadshots": 1,
            "RoundStartingEquipmentValue": 4000,
            "TeamStartingEquipmentValue": 18000
        }
    })

@app.route('/predict', methods=['POST'])
def predict_full():
    """
    Predicción completa: supervivencia + tiempo de vida
    """
    if model_package is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        # Obtener datos JSON
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        # Preprocesar datos
        X_processed, error = preprocess_input(data)
        
        if error:
            return jsonify({"error": error}), 400
        
        # Modelos a usar (mejores según entrenamiento)
        classifier = models['rf_classifier']
        regressor = models['rf_regressor']
        scaler_clf = scalers['scaler_classification']
        scaler_reg = scalers['scaler_regression']
        
        # Escalar datos
        X_clf_scaled = scaler_clf.transform(X_processed)
        X_reg_scaled = scaler_reg.transform(X_processed)
        
        # Predicciones
        # Clasificación (supervivencia)
        survival_pred = classifier.predict(X_clf_scaled)[0]
        survival_proba = classifier.predict_proba(X_clf_scaled)[0]
        
        # Regresión (tiempo de vida)
        time_pred = regressor.predict(X_reg_scaled)[0]
        
        # Construir respuesta
        response = {
            "prediction": {
                "survival": {
                    "will_survive": bool(survival_pred),
                    "probability": {
                        "death": float(survival_proba[0]),
                        "survival": float(survival_proba[1])
                    },
                    "confidence": float(max(survival_proba))
                },
                "time_alive": {
                    "predicted_seconds": float(time_pred),
                    "predicted_minutes": float(time_pred / 60),
                    "interpretation": "alto" if time_pred > 60 else "medio" if time_pred > 30 else "bajo"
                }
            },
            "input_data": data,
            "model_info": {
                "classifier": "Random Forest",
                "regressor": "Random Forest",
                "classifier_accuracy": f"{metadata['rf_clf_accuracy']:.4f}",
                "regressor_r2": f"{metadata['rf_reg_r2']:.4f}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/predict/survival', methods=['POST'])
def predict_survival():
    """Solo predicción de supervivencia"""
    if model_package is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        X_processed, error = preprocess_input(data)
        if error:
            return jsonify({"error": error}), 400
        
        classifier = models['rf_classifier']
        scaler_clf = scalers['scaler_classification']
        
        X_clf_scaled = scaler_clf.transform(X_processed)
        
        survival_pred = classifier.predict(X_clf_scaled)[0]
        survival_proba = classifier.predict_proba(X_clf_scaled)[0]
        
        return jsonify({
            "will_survive": bool(survival_pred),
            "probability": {
                "death": float(survival_proba[0]),
                "survival": float(survival_proba[1])
            },
            "confidence": float(max(survival_proba)),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/predict/time', methods=['POST'])
def predict_time():
    """Solo predicción de tiempo de vida"""
    if model_package is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        X_processed, error = preprocess_input(data)
        if error:
            return jsonify({"error": error}), 400
        
        regressor = models['rf_regressor']
        scaler_reg = scalers['scaler_regression']
        
        X_reg_scaled = scaler_reg.transform(X_processed)
        time_pred = regressor.predict(X_reg_scaled)[0]
        
        return jsonify({
            "predicted_seconds": float(time_pred),
            "predicted_minutes": float(time_pred / 60),
            "interpretation": "alto" if time_pred > 60 else "medio" if time_pred > 30 else "bajo",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predicciones en lote para múltiples jugadores"""
    if model_package is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'players' not in data:
            return jsonify({"error": "Formato requerido: {'players': [...]"}), 400
        
        players = data['players']
        
        if not isinstance(players, list):
            return jsonify({"error": "players debe ser una lista"}), 400
        
        results = []
        
        for i, player_data in enumerate(players):
            try:
                X_processed, error = preprocess_input(player_data)
                
                if error:
                    results.append({
                        "player_index": i,
                        "error": error,
                        "input_data": player_data
                    })
                    continue
                
                # Predicciones
                classifier = models['rf_classifier']
                regressor = models['rf_regressor']
                scaler_clf = scalers['scaler_classification']
                scaler_reg = scalers['scaler_regression']
                
                X_clf_scaled = scaler_clf.transform(X_processed)
                X_reg_scaled = scaler_reg.transform(X_processed)
                
                survival_pred = classifier.predict(X_clf_scaled)[0]
                survival_proba = classifier.predict_proba(X_clf_scaled)[0]
                time_pred = regressor.predict(X_reg_scaled)[0]
                
                results.append({
                    "player_index": i,
                    "will_survive": bool(survival_pred),
                    "survival_probability": float(survival_proba[1]),
                    "predicted_time_seconds": float(time_pred),
                    "input_data": player_data
                })
                
            except Exception as e:
                results.append({
                    "player_index": i,
                    "error": f"Error procesando jugador: {str(e)}",
                    "input_data": player_data
                })
        
        return jsonify({
            "results": results,
            "total_players": len(players),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en predicción por lotes: {str(e)}"}), 500

# Manejador de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado",
        "available_endpoints": [
            "/", "/health", "/features", "/predict", 
            "/predict/survival", "/predict/time", "/batch_predict"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    # Cargar modelos al iniciar
    if load_models():
        logger.info("🚀 Iniciando servidor Flask...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("❌ No se pudieron cargar los modelos. Servidor no iniciado.")
        print("\n🔧 Para solucionar:")
        print("1. Ejecuta primero 'python save_model.py' para entrenar y guardar los modelos")
        print("2. Asegúrate de que el archivo 'saved_models/csgo_model_package.pkl' existe")
        print("3. Verifica que el directorio de datos '../data/' existe con el CSV")