# test_model.py - Probar el modelo CS:GO con las características correctas
import pickle
import numpy as np
import os

print("🧪 PRUEBA DEL MODELO CS:GO")
print("=" * 40)

def load_models():
    """Cargar los modelos guardados"""
    
    model_paths = {
        'clasificacion': "models/models/modelos_clasificacion.pkl",
        'regresion': "models/models/modelos_regresion.pkl"
    }
    
    print("📂 Cargando modelos...")
    
    try:
        with open(model_paths['clasificacion'], 'rb') as f:
            modelos_clasificacion = pickle.load(f)
        
        with open(model_paths['regresion'], 'rb') as f:
            modelos_regresion = pickle.load(f)
        
        # Usar Random Forest (el más común)
        classifier = modelos_clasificacion['Random Forest']
        regressor = modelos_regresion['Random Forest']
        
        print(f"✅ Modelos cargados exitosamente")
        print(f"   🎯 Clasificador espera: {classifier.n_features_in_} características")
        print(f"   ⏱️  Regressor espera: {regressor.n_features_in_} características")
        
        return classifier, regressor
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {str(e)}")
        return None, None

def test_with_correct_features():
    """Probar el modelo con las características correctas"""
    
    classifier, regressor = load_models()
    if classifier is None or regressor is None:
        return
    
    print(f"\n🧪 Probando modelo...")
    
    # Características más importantes para CS:GO (ajustar según tu modelo)
    test_data = [
        2,      # RoundKills
        1,      # RoundHeadshots  
        1,      # RoundAssists
        15.0,   # FirstKillTime
        3500.0, # TravelledDistance
        1,      # RLethalGrenadesThrown
        0.7     # PrimaryAssaultRifle
    ]
    
    try:
        # Convertir a array numpy
        X_test = np.array(test_data).reshape(1, -1)
        
        print(f"📊 Datos de prueba: {X_test[0]}")
        
        # Predicción de supervivencia
        survival_pred = classifier.predict(X_test)[0]
        survival_proba = classifier.predict_proba(X_test)[0]
        
        print(f"\n🎯 PREDICCIÓN DE SUPERVIVENCIA:")
        print(f"   Resultado: {'✅ SOBREVIVE' if survival_pred else '❌ MUERE'}")
        print(f"   Probabilidad muerte: {survival_proba[0]:.1%}")
        print(f"   Probabilidad supervivencia: {survival_proba[1]:.1%}")
        
        # Predicción de tiempo
        time_pred = regressor.predict(X_test)[0]
        
        print(f"\n⏱️  PREDICCIÓN DE TIEMPO:")
        print(f"   Tiempo predicho: {time_pred:.1f} segundos")
        print(f"   En minutos: {time_pred/60:.2f} minutos")
        
        # Interpretación del tiempo
        if time_pred < 30:
            interpretation = "Muy corto - probablemente muere rápido"
        elif time_pred < 90:
            interpretation = "Corto - muere en la primera mitad"
        elif time_pred < 150:
            interpretation = "Medio - sobrevive hasta la mitad"
        else:
            interpretation = "Largo - sobrevive hasta el final"
            
        print(f"   Interpretación: {interpretation}")
        
        print(f"\n✅ ¡PRUEBA EXITOSA!")
        print(f"📝 El modelo funciona con {len(test_data)} características")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {str(e)}")
        print(f"💡 Ajusta las características según tu modelo entrenado")

def test_multiple_scenarios():
    """Probar múltiples escenarios"""
    
    classifier, regressor = load_models()
    if classifier is None or regressor is None:
        return
    
    print(f"\n🎮 PROBANDO MÚLTIPLES ESCENARIOS:")
    print("=" * 40)
    
    scenarios = [
        {
            "name": "Jugador Agresivo",
            "data": [4, 3, 0, 10.0, 4000.0, 2, 0.9],
            "description": "Muchos kills, headshots, poca distancia, granadas letales"
        },
        {
            "name": "Jugador Defensivo", 
            "data": [0, 0, 2, 60.0, 1500.0, 1, 0.8],
            "description": "Sin kills, solo asistencias, poca distancia, defensivo"
        },
        {
            "name": "Jugador Balanceado",
            "data": [100, 1, 1, 25.0, 500000, 1, 0.7],
            "description": "Kills moderados, balanceado en todas las estadísticas"
        }
    ]
    
    feature_names = [
        "RoundKills", "RoundHeadshots", "RoundAssists", 
        "FirstKillTime", "TravelledDistance", "RLethalGrenadesThrown", "PrimaryAssaultRifle"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Escenario {i}: {scenario['name']}")
        print(f"   📝 Descripción: {scenario['description']}")
        print(f"   📋 Datos:")
        
        # Mostrar cada característica con su valor
        for j, (feature, value) in enumerate(zip(feature_names, scenario['data'])):
            print(f"      {j+1}. {feature}: {value}")
        
        try:
            X_test = np.array(scenario['data']).reshape(1, -1)
            
            # Predicciones
            survival_pred = classifier.predict(X_test)[0]
            survival_proba = classifier.predict_proba(X_test)[0]
            time_pred = regressor.predict(X_test)[0]
            
            print(f"   🎯 Supervivencia: {'✅' if survival_pred else '❌'} ({survival_proba[1]:.1%})")
            print(f"   ⏱️  Tiempo: {time_pred:.1f}s ({time_pred/60:.1f} min)")
            
            # Interpretación del tiempo
            if time_pred < 30:
                interpretation = "Muy corto - probablemente muere rápido"
            elif time_pred < 90:
                interpretation = "Corto - muere en la primera mitad"
            elif time_pred < 150:
                interpretation = "Medio - sobrevive hasta la mitad"
            else:
                interpretation = "Largo - sobrevive hasta el final"
                
            print(f"   💡 Interpretación: {interpretation}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def main():
    """Función principal"""
    
    # Prueba básica
    test_with_correct_features()
    
    # Pruebas múltiples
    test_multiple_scenarios()
    
    print(f"\n🎉 PRUEBAS COMPLETADAS!")
    print(f"💡 Si las pruebas fallan, ajusta las características según tu modelo")

if __name__ == "__main__":
    main()