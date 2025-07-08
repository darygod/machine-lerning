# app.py - API Flask para predicciones CS:GO adaptado para tus modelos
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
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
modelos_clasificacion = None
modelos_regresion = None
scaler = None
model_info = None

# Las 20 características exactas que espera tu modelo
FEATURES_MODELO = [
    'Team', 'InternalTeamId', 'MatchId', 'RoundId', 'MatchWinner', 
    'Survived', 'AbnormalMatch', 'TimeAlive', 'TravelledDistance', 
    'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 
    'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 
    'FirstKillTime', 'RoundKills', 'RoundAssists', 'RoundHeadshots'
]

# Características para entrenamiento (sin las variables objetivo)
FEATURES_FOR_TRAINING = [f for f in FEATURES_MODELO if f not in ['Survived', 'TimeAlive']]

# Mapeo de equipos
TEAM_MAPPING = {'Terrorist': 0, 'Counter-Terrorist': 1}

def load_models():
    """Cargar modelos y componentes desde archivos guardados"""
    global modelos_clasificacion, modelos_regresion, scaler, model_info
    
    try:
        # Rutas de los archivos que guardamos
        paths = {
            'clasificacion': "models/models/modelos_clasificacion.pkl",
            'regresion': "models/models/modelos_regresion.pkl", 
            'info': "models/models/model_info.pkl",
            'scaler': "models/models/scaler.pkl"
        }
        
        # Verificar que los archivos existen
        missing_files = []
        for name, path in paths.items():
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            raise FileNotFoundError(f"Archivos faltantes: {missing_files}")
        
        # Cargar modelos
        logger.info("🔄 Cargando modelos...")
        
        with open(paths['clasificacion'], 'rb') as f:
            modelos_clasificacion = pickle.load(f)
            
        with open(paths['regresion'], 'rb') as f:
            modelos_regresion = pickle.load(f)
            
        with open(paths['info'], 'rb') as f:
            model_info = pickle.load(f)
            
        # Cargar scaler si existe
        try:
            with open(paths['scaler'], 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler cargado")
        except FileNotFoundError:
            logger.info("⚠️ Scaler no encontrado (modelo funciona sin scaler)")
            scaler = None
        
        logger.info("✅ Modelos cargados exitosamente")
        logger.info(f"📊 Características disponibles: {len(FEATURES_MODELO)}")
        logger.info(f"🔵 Modelos de clasificación: {list(modelos_clasificacion.keys())}")
        logger.info(f"🔴 Modelos de regresión: {list(modelos_regresion.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {str(e)}")
        return False

def preprocess_input(data):
    """
    Preprocesar datos de entrada exactamente como en tu código original
    
    Args:
        data (dict or list): Datos del jugador
        
    Returns:
        tuple: (X_processed, error_message)
    """
    try:
        # Convertir a DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Asegurar que todas las 20 características están presentes
        for feature in FEATURES_MODELO:
            if feature not in df.columns:
                # Valores por defecto basados en tu ejemplo
                if feature == 'Team':
                    df[feature] = 'Terrorist'
                elif feature in ['MatchWinner', 'Survived', 'AbnormalMatch']:
                    df[feature] = False
                elif feature in ['InternalTeamId', 'MatchId', 'RoundId']:
                    df[feature] = 1
                elif feature in ['TimeAlive', 'TravelledDistance', 'FirstKillTime']:
                    df[feature] = 0.0
                elif feature.startswith('Primary'):
                    df[feature] = 0.0
                else:
                    df[feature] = 0
        
        # Seleccionar características en el orden exacto
        X_pred = df[FEATURES_MODELO].copy()
        
        # Convertir Team a numérico exactamente como en tu código
        if 'Team' in X_pred.columns:
            X_pred['Team'] = X_pred['Team'].map(TEAM_MAPPING).fillna(0)
        
        # Convertir booleanos a enteros exactamente como en tu código
        bool_cols = ['MatchWinner', 'Survived', 'AbnormalMatch']
        for col in bool_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype(int)
        
        # Usar solo las 18 características para entrenamiento
        X_processed = X_pred[FEATURES_FOR_TRAINING]
        
        return X_processed, None
        
    except Exception as e:
        return None, f"Error en preprocesamiento: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    """Endpoint de inicio con información de la API"""
    if modelos_clasificacion is None or modelos_regresion is None:
        return jsonify({
            "error": "Modelos no cargados",
            "message": "Los modelos no están disponibles. Verifica que los archivos existan."
        }), 500
    
    return jsonify({
        "message": "🎮 API de Predicciones CS:GO con Tus Modelos",
        "version": "1.0.0", 
        "status": "✅ Activa",
        "models_loaded": "✅ Sí",
        "metadata": {
            "features_count": len(FEATURES_MODELO),
            "features_for_training": len(FEATURES_FOR_TRAINING),
            "classification_models": list(modelos_clasificacion.keys()),
            "regression_models": list(modelos_regresion.keys()),
            "training_date": model_info.get('fecha_entrenamiento', 'No disponible')
        },
        "endpoints": {
            "/predict": "Predicción completa (supervivencia + tiempo)",
            "/predict/survival": "Solo predicción de supervivencia", 
            "/predict/time": "Solo predicción de tiempo",
            "/batch_predict": "Predicciones en lote",
            "/health": "Estado de la API",
            "/features": "Lista de características requeridas"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud de la API"""
    if modelos_clasificacion is None or modelos_regresion is None:
        return jsonify({"status": "❌ Error", "models_loaded": False}), 500
    
    return jsonify({
        "status": "✅ Saludable",
        "models_loaded": {
            "clasificacion": len(modelos_clasificacion),
            "regresion": len(modelos_regresion)
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Obtener lista de características que acepta el modelo"""
    if modelos_clasificacion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    return jsonify({
        "required_features": FEATURES_MODELO,
        "features_for_training": FEATURES_FOR_TRAINING,
        "team_mapping": TEAM_MAPPING,
        "example_input": {
            "Team": "Terrorist",                    # 0
            "InternalTeamId": 1,                    # 1  
            "MatchId": 4,                           # 2
            "RoundId": 1,                           # 3
            "MatchWinner": True,                    # 4
            "Survived": False,                      # 5 (ignorado)
            "AbnormalMatch": False,                 # 6
            "TimeAlive": 51.12,                     # 7 (ignorado)
            "TravelledDistance": 3500.0,            # 8
            "RLethalGrenadesThrown": 1,             # 9
            "RNonLethalGrenadesThrown": 2,          # 10
            "PrimaryAssaultRifle": 0.7,             # 11
            "PrimarySniperRifle": 0.1,              # 12
            "PrimaryHeavy": 0.0,                    # 13
            "PrimarySMG": 0.1,                      # 14
            "PrimaryPistol": 0.1,                   # 15
            "FirstKillTime": 15.0,                  # 16
            "RoundKills": 2,                        # 17
            "RoundAssists": 1,                      # 18
            "RoundHeadshots": 1                     # 19
        },
        "notes": "Survived y TimeAlive son ignorados durante la predicción"
    })

@app.route('/predict', methods=['POST'])
def predict_full():
    """
    Predicción completa: supervivencia + tiempo de vida
    """
    if modelos_clasificacion is None or modelos_regresion is None:
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
        
        # Usar Random Forest como modelos principales (como en tu código)
        classifier = modelos_clasificacion['Random Forest']
        regressor = modelos_regresion['Random Forest']
        
        # Hacer predicciones (sin scaler como en tu código original)
        # Si tu modelo necesita scaler, descomenta las siguientes líneas:
        # if scaler:
        #     X_processed = scaler.transform(X_processed)
        
        # Predicción de clasificación (supervivencia)
        survival_pred = classifier.predict(X_processed)[0]
        survival_proba = classifier.predict_proba(X_processed)[0]
        
        # Predicción de regresión (tiempo de vida)
        time_pred = regressor.predict(X_processed)[0]
        
        # Construir respuesta en formato similar a tu estructura original
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
                "features_used": len(FEATURES_FOR_TRAINING),
                "note": "Basado en tu modelo CRISP-DM"
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
    if modelos_clasificacion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        X_processed, error = preprocess_input(data)
        if error:
            return jsonify({"error": error}), 400
        
        classifier = modelos_clasificacion['Random Forest']
        
        survival_pred = classifier.predict(X_processed)[0]
        survival_proba = classifier.predict_proba(X_processed)[0]
        
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
    if modelos_regresion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        X_processed, error = preprocess_input(data)
        if error:
            return jsonify({"error": error}), 400
        
        regressor = modelos_regresion['Random Forest']
        time_pred = regressor.predict(X_processed)[0]
        
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
    if modelos_clasificacion is None or modelos_regresion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'players' not in data:
            return jsonify({"error": "Formato requerido: {'players': [...]}"}), 400
        
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
                
                # Predicciones con Random Forest
                classifier = modelos_clasificacion['Random Forest']
                regressor = modelos_regresion['Random Forest']
                
                survival_pred = classifier.predict(X_processed)[0]
                survival_proba = classifier.predict_proba(X_processed)[0]
                time_pred = regressor.predict(X_processed)[0]
                
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
        logger.info("🚀 Iniciando servidor Flask en puerto 5000...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("❌ No se pudieron cargar los modelos. Servidor no iniciado.")
        print("\n🔧 Para solucionar:")
        print("1. Ejecuta tu código CRISP-DM completo para entrenar los modelos")
        print("2. Ejecuta el código de guardado para crear los archivos .pkl")
        print("3. Asegúrate de que existan estos archivos:")
        print("   - models/modelos_clasificacion.pkl")
        print("   - models/modelos_regresion.pkl") 
        print("   - models/model_info.pkl")
        print("   - models/scaler.pkl (opcional)")