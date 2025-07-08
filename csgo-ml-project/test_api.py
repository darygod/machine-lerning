# test_api.py - Ejemplos de uso de la API CS:GO
import requests
import json

# URL base de la API (cambiar si está en otro servidor)
BASE_URL = "http://localhost:5001"

def test_api_connection():
    """Probar que la API está funcionando"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API conectada exitosamente!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Error conectando a la API: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar a la API. ¿Está ejecutándose?")
        return False

def test_health():
    """Verificar estado de salud de la API"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"\n🔍 Estado de salud: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

def get_features():
    """Obtener características que acepta el modelo"""
    try:
        response = requests.get(f"{BASE_URL}/features")
        if response.status_code == 200:
            print("\n📋 Características del modelo:")
            data = response.json()
            print(f"Características requeridas: {len(data['required_features'])}")
            print(f"Todas las características: {len(data['all_features'])}")
            print("\n💡 Ejemplo de entrada:")
            print(json.dumps(data['example_input'], indent=2))
            return data['example_input']
        else:
            print(f"Error obteniendo características: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_survival_prediction():
    """Probar predicción de supervivencia"""
    print("\n🎯 Probando predicción de supervivencia...")
    
    # Datos de ejemplo - jugador con buenas estadísticas
    player_data = {
        "Team": "Terrorist",
        "InternalTeamId": 1,
        "MatchId": 100,
        "RoundId": 5,
        "MatchWinner": True,
        "TravelledDistance": 3500.0,
        "RLethalGrenadesThrown": 1,
        "RNonLethalGrenadesThrown": 2,
        "PrimaryAssaultRifle": 0.8,
        "PrimarySniperRifle": 0.0,
        "PrimaryHeavy": 0.0,
        "PrimarySMG": 0.1,
        "PrimaryPistol": 0.1,
        "FirstKillTime": 15.0,
        "RoundKills": 3,
        "RoundAssists": 2,
        "RoundHeadshots": 2,
        "RoundStartingEquipmentValue": 4500,
        "TeamStartingEquipmentValue": 20000
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/survival",
            json=player_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"¿Sobrevive?: {'✅ SÍ' if result['will_survive'] else '❌ NO'}")
            print(f"Probabilidad de supervivencia: {result['probability']['survival']:.1%}")
            print(f"Confianza: {result['confidence']:.1%}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {e}")

def test_time_prediction():
    """Probar predicción de tiempo de vida"""
    print("\n⏱️ Probando predicción de tiempo de vida...")
    
    # Datos de ejemplo - jugador defensivo
    player_data = {
        "Team": "Counter-Terrorist",
        "InternalTeamId": 2,
        "MatchId": 200,
        "RoundId": 8,
        "MatchWinner": False,
        "TravelledDistance": 2000.0,
        "RLethalGrenadesThrown": 0,
        "RNonLethalGrenadesThrown": 1,
        "PrimaryAssaultRifle": 0.9,
        "PrimarySniperRifle": 0.1,
        "PrimaryHeavy": 0.0,
        "PrimarySMG": 0.0,
        "PrimaryPistol": 0.0,
        "FirstKillTime": 45.0,
        "RoundKills": 1,
        "RoundAssists": 1,
        "RoundHeadshots": 1,
        "RoundStartingEquipmentValue": 5000,
        "TeamStartingEquipmentValue": 22000
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/time",
            json=player_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Tiempo predicho: {result['predicted_seconds']:.1f} segundos")
            print(f"En minutos: {result['predicted_minutes']:.2f} minutos")
            print(f"Interpretación: {result['interpretation']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {e}")

def test_full_prediction():
    """Probar predicción completa (supervivencia + tiempo)"""
    print("\n🎮 Probando predicción completa...")
    
    # Datos de ejemplo - jugador balanceado
    player_data = {
        "Team": "Terrorist",
        "InternalTeamId": 1,
        "MatchId": 300,
        "RoundId": 12,
        "MatchWinner": True,
        "TravelledDistance": 3000.0,
        "RLethalGrenadesThrown": 1,
        "RNonLethalGrenadesThrown": 1,
        "PrimaryAssaultRifle": 0.6,
        "PrimarySniperRifle": 0.2,
        "PrimaryHeavy": 0.0,
        "PrimarySMG": 0.1,
        "PrimaryPistol": 0.1,
        "FirstKillTime": 25.0,
        "RoundKills": 2,
        "RoundAssists": 1,
        "RoundHeadshots": 1,
        "RoundStartingEquipmentValue": 4000,
        "TeamStartingEquipmentValue": 18000
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=player_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extraer resultados
            survival = result['prediction']['survival']
            time_info = result['prediction']['time_alive']
            
            print("📊 RESULTADOS COMPLETOS:")
            print(f"   ¿Sobrevive?: {'✅ SÍ' if survival['will_survive'] else '❌ NO'}")
            print(f"   Probabilidad: {survival['probability']['survival']:.1%}")
            print(f"   Confianza: {survival['confidence']:.1%}")
            print(f"   Tiempo predicho: {time_info['predicted_seconds']:.1f}s ({time_info['predicted_minutes']:.2f} min)")
            print(f"   Rendimiento: {time_info['interpretation']}")
            
            print(f"\n🤖 Información del modelo:")
            model_info = result['model_info']
            print(f"   Clasificador: {model_info['classifier']} (Accuracy: {model_info['classifier_accuracy']})")
            print(f"   Regresor: {model_info['regressor']} (R²: {model_info['regressor_r2']})")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {e}")

def test_batch_prediction():
    """Probar predicciones en lote"""
    print("\n📦 Probando predicciones en lote...")
    
    # Múltiples jugadores con diferentes perfiles
    batch_data = {
        "players": [
            {
                "Team": "Terrorist",
                "RoundKills": 4,
                "RoundHeadshots": 3,
                "TravelledDistance": 4000.0,
                "PrimaryAssaultRifle": 0.9,
                "RoundStartingEquipmentValue": 5000
            },
            {
                "Team": "Counter-Terrorist", 
                "RoundKills": 0,
                "RoundHeadshots": 0,
                "TravelledDistance": 1000.0,
                "PrimaryPistol": 0.8,
                "RoundStartingEquipmentValue": 800
            },
            {
                "Team": "Terrorist",
                "RoundKills": 2,
                "RoundHeadshots": 1,
                "TravelledDistance": 2500.0,
                "PrimarySniperRifle": 0.7,
                "RoundStartingEquipmentValue": 4750
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📊 Procesados: {result['total_players']} jugadores")
            print(f"✅ Exitosos: {result['successful_predictions']}")
            
            for player_result in result['results']:
                if 'error' not in player_result:
                    print(f"\n   Jugador {player_result['player_index']}:")
                    print(f"      Supervivencia: {'✅' if player_result['will_survive'] else '❌'} ({player_result['survival_probability']:.1%})")
                    print(f"      Tiempo: {player_result['predicted_time_seconds']:.1f}s")
                else:
                    print(f"\n   Jugador {player_result['player_index']}: ❌ {player_result['error']}")
                    
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Ejecutar todas las pruebas"""
    print("🧪 PRUEBAS DE LA API CS:GO")
    print("=" * 50)
    
    # Verificar conexión
    if not test_api_connection():
        print("\n❌ No se puede conectar a la API. Asegúrate de que esté ejecutándose:")
        print("   python app.py")
        return
    
    # Pruebas individuales
    test_health()
    get_features()
    test_survival_prediction()
    test_time_prediction()
    test_full_prediction()
    test_batch_prediction()
    
    print("\n✅ Todas las pruebas completadas!")
    print("\n💡 Casos de uso:")
    print("   - Usar /predict para análisis completo")
    print("   - Usar /predict/survival para decisiones rápidas")
    print("   - Usar /predict/time para estrategias de tiempo")
    print("   - Usar /batch_predict para análisis de equipos")

if __name__ == "__main__":
    main()