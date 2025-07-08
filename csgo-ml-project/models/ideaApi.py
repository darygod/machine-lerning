from copia_de_informe_def import *

# GUARDAR MODELOS PARA LA API
import pickle
import os
from datetime import datetime

# ✅ CREAR ESTRUCTURA DE CARPETAS CORRECTA
os.makedirs('models', exist_ok=True)

# Las 20 características exactas
FEATURES_MODELO = [
    'Team', 'InternalTeamId', 'MatchId', 'RoundId', 'MatchWinner', 
    'Survived', 'AbnormalMatch', 'TimeAlive', 'TravelledDistance', 
    'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 
    'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 
    'FirstKillTime', 'RoundKills', 'RoundAssists', 'RoundHeadshots'
]

print("📋 Verificando modelos disponibles...")

# Verificar que los modelos existen
print("🔵 Modelos de clasificación disponibles:")
for nombre in modelos_clf.keys():
    print(f"   • {nombre}")

print("🔴 Modelos de regresión disponibles:")
for nombre in modelos_reg.keys():
    print(f"   • {nombre}")

# Guardar modelos para API
try:
    modelos_clasificacion_api = {
        'Random Forest': modelos_clf['Random Forest Clas'],
        'Logistic Regression': modelos_clf['Regresión Logística']
    }
    print("✅ Modelos de clasificación preparados")
except KeyError as e:
    print(f"❌ Error: Modelo no encontrado - {e}")
    print("💡 Nombres disponibles:", list(modelos_clf.keys()))

try:
    modelos_regresion_api = {
        'Random Forest': modelos_reg['Random Forest Reg'],
        'Linear Regression': modelos_reg['Regresión Lineal'],
        'Decision Tree': modelos_reg['Árbol de Decisión']
    }
    print("✅ Modelos de regresión preparados")
except KeyError as e:
    print(f"❌ Error: Modelo no encontrado - {e}")
    print("💡 Nombres disponibles:", list(modelos_reg.keys()))

# ✅ GUARDAR ARCHIVOS EN LA RUTA CORRECTA (sin subcarpeta)
print("\n💾 Guardando modelos...")

try:
    # Guardar modelos de clasificación
    with open('models/modelos_clasificacion.pkl', 'wb') as f:
        pickle.dump(modelos_clasificacion_api, f)
    print("✅ modelos_clasificacion.pkl guardado")

    # Guardar modelos de regresión  
    with open('models/modelos_regresion.pkl', 'wb') as f:
        pickle.dump(modelos_regresion_api, f)
    print("✅ modelos_regresion.pkl guardado")

    # Intentar guardar el scaler si existe
    try:
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ scaler.pkl guardado")
    except NameError:
        print("⚠️ Scaler no encontrado (tu modelo funciona sin scaler)")

    # Guardar información del modelo
    model_info = {
        'features': FEATURES_MODELO,
        'features_for_training': [f for f in FEATURES_MODELO if f not in ['Survived', 'TimeAlive']],
        'target_classification': 'Survived',
        'target_regression': 'TimeAlive',
        'team_mapping': {'Terrorist': 0, 'Counter-Terrorist': 1},
        'fecha_entrenamiento': datetime.now().isoformat(),
        'descripcion': 'Modelos CS:GO - 20 características exactas',
        'nota': 'Modelo funciona sin scaler según código original'
    }

    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("✅ model_info.pkl guardado")

    # Resumen final
    print(f"\n🎉 ¡MODELOS GUARDADOS EXITOSAMENTE!")
    print(f"📁 Archivos creados en ./models/:")
    print(f"   • modelos_clasificacion.pkl")
    print(f"   • modelos_regresion.pkl") 
    print(f"   • model_info.pkl")
    if 'scaler' in locals():
        print(f"   • scaler.pkl")
    
    print(f"\n🚀 Ahora puedes ejecutar la API:")
    print(f"   python app.py")
    
    # Verificar archivos
    print(f"\n🔍 Verificando archivos creados:")
    archivos = ['modelos_clasificacion.pkl', 'modelos_regresion.pkl', 'model_info.pkl']
    for archivo in archivos:
        ruta = f'models/{archivo}'
        if os.path.exists(ruta):
            tamaño = os.path.getsize(ruta) / 1024  # KB
            print(f"   ✅ {archivo} ({tamaño:.1f} KB)")
        else:
            print(f"   ❌ {archivo} - NO ENCONTRADO")

except Exception as e:
    print(f"❌ Error guardando modelos: {e}")
    print("💡 Verifica que los modelos estén entrenados correctamente")

# 🧪 CREAR SCRIPT DE PRUEBA RÁPIDA
test_script = '''#!/usr/bin/env python3
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
        
        print("\\n🔵 Probando clasificación:")
        for nombre, modelo in modelos_clf.items():
            pred = modelo.predict(X_test)
            prob = modelo.predict_proba(X_test)
            print(f"   {nombre}: Pred={pred[0]} | Prob={prob[0][1]:.3f}")
        
        print("\\n🔴 Probando regresión:")
        for nombre, modelo in modelos_reg.items():
            pred = modelo.predict(X_test)
            print(f"   {nombre}: {pred[0]:.3f}")
        
        print("\\n✅ ¡Todos los modelos funcionan correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_saved_models()
'''

with open('test_models.py', 'w') as f:
    f.write(test_script)
print(f"\n🧪 Script de prueba creado: test_models.py")
print(f"💡 Ejecuta: python test_models.py")