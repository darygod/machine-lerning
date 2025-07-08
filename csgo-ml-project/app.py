# app_fixed.py - API Flask corregida para usar solo 7 características
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

# ⭐ CARACTERÍSTICAS CORREGIDAS - Solo las 7 que espera tu modelo
# Basándome en el error, tu modelo fue entrenado con estas características principales:
FEATURES_MODELO_CORREGIDAS = [
    'RoundKills',           # 0 - Kills en el round
    'RoundHeadshots',       # 1 - Headshots en el round  
    'RoundAssists',         # 2 - Asistencias en el round
    'FirstKillTime',        # 3 - Tiempo del primer kill
    'TravelledDistance',    # 4 - Distancia recorrida
    'RLethalGrenadesThrown', # 5 - Granadas letales lanzadas
    'PrimaryAssaultRifle'   # 6 - Proporción de uso de rifle de asalto
]

# Todas las características originales para compatibilidad con la entrada
FEATURES_COMPLETAS = [
    'Team', 'InternalTeamId', 'MatchId', 'RoundId', 'MatchWinner', 
    'Survived', 'AbnormalMatch', 'TimeAlive', 'TravelledDistance', 
    'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 
    'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 
    'FirstKillTime', 'RoundKills', 'RoundAssists', 'RoundHeadshots'
]

def load_models():
    """Cargar modelos y componentes desde archivos guardados"""
    global modelos_clasificacion, modelos_regresion, scaler, model_info
    
    try:
        # Rutas de los archivos
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
        
        # Verificar dimensiones del modelo
        rf_classifier = modelos_clasificacion['Random Forest']
        expected_features = rf_classifier.n_features_in_
        
        logger.info("✅ Modelos cargados exitosamente")
        logger.info(f"🎯 Modelo espera: {expected_features} características")
        logger.info(f"📊 Características a usar: {len(FEATURES_MODELO_CORREGIDAS)}")
        logger.info(f"🔵 Modelos de clasificación: {list(modelos_clasificacion.keys())}")
        logger.info(f"🔴 Modelos de regresión: {list(modelos_regresion.keys())}")
        
        # Verificar compatibilidad
        if expected_features != len(FEATURES_MODELO_CORREGIDAS):
            logger.warning(f"⚠️ ADVERTENCIA: Modelo espera {expected_features} características, "
                    f"pero definimos {len(FEATURES_MODELO_CORREGIDAS)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {str(e)}")
        return False

def preprocess_input(data):
    """
    Preprocesar datos de entrada - VERSIÓN CORREGIDA
    Extrae solo las 7 características que necesita el modelo
    
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
        
        # Asegurar que todas las características completas están presentes (para compatibilidad)
        for feature in FEATURES_COMPLETAS:
            if feature not in df.columns:
                # Valores por defecto
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
        
        # ⭐ CAMBIO PRINCIPAL: Extraer solo las 7 características que necesita el modelo
        X_reduced = pd.DataFrame()
        
        for feature in FEATURES_MODELO_CORREGIDAS:
            if feature in df.columns:
                X_reduced[feature] = df[feature]
            else:
                # Valor por defecto si falta la característica
                if feature in ['FirstKillTime', 'TravelledDistance', 'PrimaryAssaultRifle']:
                    X_reduced[feature] = 0.0
                else:
                    X_reduced[feature] = 0
        
        # Convertir tipos de datos apropiados
        numeric_features = ['FirstKillTime', 'TravelledDistance', 'PrimaryAssaultRifle']
        for feature in numeric_features:
            if feature in X_reduced.columns:
                X_reduced[feature] = pd.to_numeric(X_reduced[feature], errors='coerce').fillna(0.0)
        
        integer_features = ['RoundKills', 'RoundHeadshots', 'RoundAssists', 'RLethalGrenadesThrown']
        for feature in integer_features:
            if feature in X_reduced.columns:
                X_reduced[feature] = pd.to_numeric(X_reduced[feature], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"📊 Datos procesados: {X_reduced.shape} - Características: {list(X_reduced.columns)}")
        
        return X_reduced, None
        
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
    
    # Obtener información del modelo
    rf_classifier = modelos_clasificacion['Random Forest']
    expected_features = rf_classifier.n_features_in_
    
    return jsonify({
        "message": "🎮 API de Predicciones CS:GO - VERSIÓN CORREGIDA",
        "version": "1.1.0", 
        "status": "✅ Activa",
        "models_loaded": "✅ Sí",
        "fix_applied": "✅ Corregido para usar 7 características",
        "metadata": {
            "model_expects": expected_features,
            "features_used": len(FEATURES_MODELO_CORREGIDAS),
            "features_list": FEATURES_MODELO_CORREGIDAS,
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
    
    rf_classifier = modelos_clasificacion['Random Forest']
    expected_features = rf_classifier.n_features_in_
    
    return jsonify({
        "status": "✅ Saludable",
        "models_loaded": {
            "clasificacion": len(modelos_clasificacion),
            "regresion": len(modelos_regresion)
        },
        "model_compatibility": {
            "expected_features": expected_features,
            "api_features": len(FEATURES_MODELO_CORREGIDAS),
            "compatible": expected_features == len(FEATURES_MODELO_CORREGIDAS)
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Obtener lista de características que acepta el modelo"""
    if modelos_clasificacion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    rf_classifier = modelos_clasificacion['Random Forest']
    expected_features = rf_classifier.n_features_in_
    
    return jsonify({
        "model_expects": expected_features,
        "required_features": FEATURES_MODELO_CORREGIDAS,
        "optional_features": [f for f in FEATURES_COMPLETAS if f not in FEATURES_MODELO_CORREGIDAS],
        "example_input": {
            "RoundKills": 2,               # Kills en este round
            "RoundHeadshots": 1,           # Headshots en este round
            "RoundAssists": 1,             # Asistencias en este round
            "FirstKillTime": 15.0,         # Tiempo del primer kill (segundos)
            "TravelledDistance": 3500.0,   # Distancia recorrida (unidades)
            "RLethalGrenadesThrown": 1,    # Granadas letales lanzadas
            "PrimaryAssaultRifle": 0.7     # Proporción de uso de assault rifle
        },
        "example_input_full": {
            # Puedes incluir todas las características originales para compatibilidad
            "Team": "Terrorist",
            "InternalTeamId": 1,
            "MatchId": 4,
            "RoundId": 1,
            "MatchWinner": True,
            "Survived": False,  # Será ignorado
            "AbnormalMatch": False,
            "TimeAlive": 51.12, # Será ignorado
            "TravelledDistance": 3500.0,   # ✅ USADO
            "RLethalGrenadesThrown": 1,    # ✅ USADO
            "RNonLethalGrenadesThrown": 2,
            "PrimaryAssaultRifle": 0.7,    # ✅ USADO
            "PrimarySniperRifle": 0.1,
            "PrimaryHeavy": 0.0,
            "PrimarySMG": 0.1,
            "PrimaryPistol": 0.1,
            "FirstKillTime": 15.0,         # ✅ USADO
            "RoundKills": 2,               # ✅ USADO
            "RoundAssists": 1,             # ✅ USADO
            "RoundHeadshots": 1            # ✅ USADO
        },
        "notes": "Solo se usan 7 características específicas, las demás son ignoradas"
    })

@app.route('/predict', methods=['POST'])
def predict_full():
    """
    Predicción completa: supervivencia + tiempo de vida - VERSIÓN CORREGIDA
    """
    if modelos_clasificacion is None or modelos_regresion is None:
        return jsonify({"error": "Modelos no cargados"}), 500
    
    try:
        # Obtener datos JSON
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
        
        # Preprocesar datos - AHORA USANDO SOLO 7 CARACTERÍSTICAS
        X_processed, error = preprocess_input(data)
        
        if error:
            return jsonify({"error": error}), 400
        
        # Verificar dimensiones
        logger.info(f"📊 Forma de datos procesados: {X_processed.shape}")
        
        # Usar Random Forest como modelos principales
        classifier = modelos_clasificacion['Random Forest']
        regressor = modelos_regresion['Random Forest']
        
        # Convertir a numpy array
        X_array = X_processed.values
        
        # Hacer predicciones
        survival_pred = classifier.predict(X_array)[0]
        survival_proba = classifier.predict_proba(X_array)[0]
        time_pred = regressor.predict(X_array)[0]
        
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
            "processed_features": {
                "used_features": FEATURES_MODELO_CORREGIDAS,
                "feature_values": X_processed.iloc[0].to_dict()
            },
            "model_info": {
                "classifier": "Random Forest",
                "regressor": "Random Forest",
                "features_used": len(FEATURES_MODELO_CORREGIDAS),
                "model_version": "Corregido v1.1",
                "note": "Usando solo las 7 características principales"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Predicción exitosa: Supervivencia={survival_pred}, Tiempo={time_pred:.1f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/predict/survival', methods=['POST'])
def predict_survival():
    """Solo predicción de supervivencia - VERSIÓN CORREGIDA"""
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
        X_array = X_processed.values
        
        survival_pred = classifier.predict(X_array)[0]
        survival_proba = classifier.predict_proba(X_array)[0]
        
        return jsonify({
            "will_survive": bool(survival_pred),
            "probability": {
                "death": float(survival_proba[0]),
                "survival": float(survival_proba[1])
            },
            "confidence": float(max(survival_proba)),
            "features_used": FEATURES_MODELO_CORREGIDAS,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/predict/time', methods=['POST'])
def predict_time():
    """Solo predicción de tiempo de vida - VERSIÓN CORREGIDA"""
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
        X_array = X_processed.values
        time_pred = regressor.predict(X_array)[0]
        
        return jsonify({
            "predicted_seconds": float(time_pred),
            "predicted_minutes": float(time_pred / 60),
            "interpretation": "alto" if time_pred > 60 else "medio" if time_pred > 30 else "bajo",
            "features_used": FEATURES_MODELO_CORREGIDAS,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predicciones en lote - VERSIÓN CORREGIDA"""
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
                
                X_array = X_processed.values
                
                survival_pred = classifier.predict(X_array)[0]
                survival_proba = classifier.predict_proba(X_array)[0]
                time_pred = regressor.predict(X_array)[0]
                
                results.append({
                    "player_index": i,
                    "will_survive": bool(survival_pred),
                    "survival_probability": float(survival_proba[1]),
                    "predicted_time_seconds": float(time_pred),
                    "features_used": FEATURES_MODELO_CORREGIDAS,
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
            "model_version": "Corregido v1.1",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en predicción por lotes: {str(e)}"}), 500

# Manejadores de errores
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
        logger.info("🚀 Iniciando servidor Flask CORREGIDO en puerto 5001...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("❌ No se pudieron cargar los modelos. Servidor no iniciado.")
        print("\n🔧 Para solucionar:")
        print("1. Ejecuta 'python model_diagnostic.py' para diagnosticar el problema")
        print("2. Verifica que los modelos están entrenados correctamente")
        print("3. Asegúrate de que existan estos archivos:")
        print("   - models/modelos_clasificacion.pkl")
        print("   - models/modelos_regresion.pkl") 
        print("   - models/model_info.pkl")