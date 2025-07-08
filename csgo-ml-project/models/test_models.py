#!/usr/bin/env python3
"""Prueba rápida de los modelos guardados"""
import pickle
import numpy as np

def test_saved_models():
    print("🧪 PROBANDO MODELOS GUARDADOS")
    print("=" * 30)
    
    try:
        # Cargar modelos
        with open('models/modelos_clasificacion.pkl', 'rb') as f:
            modelos_clf = pickle.load(f)
        print("✅ Modelos de clasificación cargados")
        
        with open('models/modelos_regresion.pkl', 'rb') as f:
            modelos_reg = pickle.load(f)
        print("✅ Modelos de regresión cargados")
        
        with open('models/model_info.pkl', 'rb') as f:
            info = pickle.load(f)
        print("✅ Información del modelo cargada")
        
        # Datos de prueba (las 18 características sin Survived y TimeAlive)
        features_training = info['features_for_training']
        print(f"📊 Características para entrenamiento: {len(features_training)}")
        
        # Crear datos sintéticos para probar
        np.random.seed(42)
        X_test = np.random.randn(3, len(features_training))
        
        print("\n🔵 Probando clasificación:")
        for nombre, modelo in modelos_clf.items():
            pred = modelo.predict(X_test)
            prob = modelo.predict_proba(X_test)
            print(f"   {nombre}: Pred={pred[0]} | Prob={prob[0][1]:.3f}")
        
        print("\n🔴 Probando regresión:")
        for nombre, modelo in modelos_reg.items():
            pred = modelo.predict(X_test)
            print(f"   {nombre}: {pred[0]:.3f}")
        
        print("\n✅ ¡Todos los modelos funcionan correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_saved_models()
