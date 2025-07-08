from copia_de_informe_def import *

# SOLUCIÓN FINAL - Las 20 características exactas que el modelo espera
nuevo_jugador = pd.DataFrame([{
    'Team': 'Terrorist',                    # 0
    'InternalTeamId': 1,                    # 1  
    'MatchId': 4,                           # 2
    'RoundId': 1,                           # 3
    'MatchWinner': True,                    # 4
    'Survived': False,                      # 5
    'AbnormalMatch': False,                 # 6
    'TimeAlive': 51.12,                     # 7
    'TravelledDistance': 3500.0,            # 8
    'RLethalGrenadesThrown': 1,             # 9
    'RNonLethalGrenadesThrown': 2,          # 10
    'PrimaryAssaultRifle': 0.7,             # 11
    'PrimarySniperRifle': 0.1,              # 12
    'PrimaryHeavy': 0.0,                    # 13
    'PrimarySMG': 0.1,                      # 14
    'PrimaryPistol': 0.1,                   # 15
    'FirstKillTime': 15.0,                  # 16
    'RoundKills': 2,                        # 17
    'RoundAssists': 1,                      # 18
    'RoundHeadshots': 1                     # 19
}])

# Características en el orden exacto que espera el modelo
FEATURES_MODELO = [
    'Team', 'InternalTeamId', 'MatchId', 'RoundId', 'MatchWinner', 
    'Survived', 'AbnormalMatch', 'TimeAlive', 'TravelledDistance', 
    'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 
    'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 
    'FirstKillTime', 'RoundKills', 'RoundAssists', 'RoundHeadshots'
]

# Seleccionar características en el orden correcto
X_pred = nuevo_jugador[FEATURES_MODELO]

# Convertir Team a numérico y booleanos a enteros
X_pred = X_pred.copy()
X_pred['Team'] = X_pred['Team'].map({'Terrorist': 0, 'Counter-Terrorist': 1})
X_pred['MatchWinner'] = X_pred['MatchWinner'].astype(int)
X_pred['Survived'] = X_pred['Survived'].astype(int)
X_pred['AbnormalMatch'] = X_pred['AbnormalMatch'].astype(int)

# Hacer predicción SIN scaler (el modelo puede funcionar con datos sin escalar)
try:
    supervivencia = modelos_entrenados['clasificacion']['Random Forest'].predict(X_pred)[0]
    probabilidad = modelos_entrenados['clasificacion']['Random Forest'].predict_proba(X_pred)[0]
    tiempo = modelos_entrenados['regresion']['Random Forest'].predict(X_pred)[0]
    
    print("🎯 RESULTADOS:")
    print(f"¿Sobrevive?: {'✅ SÍ' if supervivencia == 1 else '❌ NO'}")
    print(f"Probabilidad: {probabilidad[1]:.1%}")
    print(f"Tiempo: {tiempo:.1f}s")
    
except Exception as e:
    print(f"Error: {e}")
    print("💡 El modelo puede necesitar datos escalados de manera diferente")