# save_model.py - Versión corregida que maneja valores NaN
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
import os
import warnings

# Suprimir warnings de tipos mixtos
warnings.filterwarnings('ignore')

def entrenar_y_guardar_modelos():
    """
    Entrena los modelos de CS:GO y los guarda para uso en API
    Versión mejorada que maneja valores NaN correctamente
    """
    print("🚀 Iniciando entrenamiento y guardado de modelos...")
    
    # 1. Cargar datos con configuración robusta
    print("📊 Cargando datos...")
    file_path = "data/Anexo ET_demo_round_traces_2022.csv"
    
    try:
        # Cargar con configuración robusta para manejar tipos mixtos
        df = pd.read_csv(
            file_path, 
            delimiter=';',
            low_memory=False,  # Evita el warning de tipos mixtos
            na_values=['', 'NULL', 'null', 'NaN', 'nan', '-', 'N/A']  # Valores que deben ser tratados como NaN
        )
        print(f"✅ Datos cargados: {df.shape}")
        
    except FileNotFoundError:
        print("❌ Archivo no encontrado. Intentando con ruta del directorio actual...")
        file_path = "./Anexo ET_demo_round_traces_2022.csv"
        df = pd.read_csv(
            file_path, 
            delimiter=';',
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NaN', 'nan', '-', 'N/A']
        )
        print(f"✅ Datos cargados desde directorio actual: {df.shape}")
    
    # 2. Diagnóstico inicial de datos
    print(f"\n🔍 DIAGNÓSTICO INICIAL DE DATOS")
    print(f"   📊 Forma del dataset: {df.shape}")
    print(f"   📋 Columnas: {df.columns.tolist()}")
    print(f"   🔢 Tipos de datos únicos: {df.dtypes.value_counts().to_dict()}")
    print(f"   ❓ Total valores nulos: {df.isnull().sum().sum()}")
    
    # Mostrar columnas con más valores nulos
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    cols_with_missing = missing_summary[missing_summary > 0]
    
    if len(cols_with_missing) > 0:
        print(f"\n📊 Columnas con valores faltantes:")
        for col, missing_count in cols_with_missing.head(10).items():
            percentage = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count} ({percentage:.1f}%)")
    
    # 3. Limpieza inicial
    print(f"\n🧹 LIMPIEZA INICIAL DE DATOS")
    
    # Eliminar columna índice si existe
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        print("   ✅ Eliminada columna 'Unnamed: 0'")
    
    # Verificar columnas críticas
    critical_columns = ['Team', 'MatchWinner', 'TimeAlive', 'Survived']
    missing_critical = [col for col in critical_columns if col not in df.columns]
    
    if missing_critical:
        print(f"❌ Faltan columnas críticas: {missing_critical}")
        print(f"📋 Columnas disponibles: {df.columns.tolist()}")
        return None
    
    # 4. Conversión y limpieza de tipos de datos
    print(f"\n🔧 CONVERSIÓN DE TIPOS DE DATOS")
    
    # Convertir columnas numéricas problemáticas
    numeric_columns = ['TimeAlive', 'TravelledDistance', 'FirstKillTime']
    
    for col in numeric_columns:
        if col in df.columns:
            print(f"   🔄 Procesando {col}...")
            
            # Convertir a string primero para limpiar
            df[col] = df[col].astype(str)
            
            # Limpiar caracteres problemáticos
            df[col] = df[col].str.replace(',', '.')  # Comas por puntos
            df[col] = df[col].str.replace(' ', '')   # Eliminar espacios
            df[col] = df[col].replace(['nan', 'NaN', 'NULL', ''], np.nan)  # Estandarizar NaN
            
            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Reportar estadísticas
            non_null_count = df[col].count()
            print(f"      ✅ {col}: {non_null_count}/{len(df)} valores válidos")
    
    # Verificar columnas booleanas/categóricas
    categorical_columns = ['Team', 'MatchWinner', 'Survived', 'AbnormalMatch']
    
    for col in categorical_columns:
        if col in df.columns:
            print(f"   🏷️  {col}: {df[col].value_counts().to_dict()}")
    
    # 5. Selección de características robusta
    print(f"\n🎯 SELECCIÓN DE CARACTERÍSTICAS")
    
    # Características principales que normalmente están disponibles
    base_features = [
        'InternalTeamId', 'MatchId', 'RoundId', 'TravelledDistance',
        'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown',
        'RoundKills', 'RoundAssists', 'RoundHeadshots',
        'FirstKillTime'
    ]
    
    # Características adicionales opcionales
    optional_features = [
        'PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy',
        'PrimarySMG', 'PrimaryPistol', 'MatchKills', 'MatchAssists',
        'MatchHeadshots', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue'
    ]
    
    # Encontrar características disponibles
    available_features = []
    
    for feature_list, list_name in [(base_features, "base"), (optional_features, "optional")]:
        for feature in feature_list:
            if feature in df.columns:
                # Verificar que la columna tenga datos útiles
                non_null_ratio = df[feature].count() / len(df)
                if non_null_ratio > 0.5:  # Al menos 50% de datos no nulos
                    available_features.append(feature)
                    print(f"   ✅ {feature} ({list_name}) - {non_null_ratio:.1%} datos válidos")
                else:
                    print(f"   ⚠️ {feature} ({list_name}) - Solo {non_null_ratio:.1%} datos válidos, excluido")
    
    print(f"\n📋 Características seleccionadas: {len(available_features)}")
    
    # 6. Crear variables objetivo
    print(f"\n🎯 CREANDO VARIABLES OBJETIVO")
    
    df_model = df.copy()
    
    # Variable de clasificación: Supervivencia
    if 'Survived' in df_model.columns:
        # Limpiar la columna Survived
        df_model['Survived'] = df_model['Survived'].fillna(False)  # Asumir que NaN = no sobrevivió
        y_classification = df_model['Survived'].astype(int)
        print(f"   ✅ Usando columna 'Survived' para clasificación")
        print(f"   📊 Distribución: {y_classification.value_counts().to_dict()}")
    else:
        # Crear basada en TimeAlive
        threshold = df_model['TimeAlive'].median()
        y_classification = (df_model['TimeAlive'] > threshold).astype(int)
        print(f"   ✅ Creando supervivencia basada en TimeAlive > {threshold:.2f}")
        print(f"   📊 Distribución: {y_classification.value_counts().to_dict()}")
    
    # Variable de regresión: TimeAlive para sobrevivientes
    survival_mask = (df_model['TimeAlive'] > 0) & (df_model['TimeAlive'].notna())
    y_regression = df_model.loc[survival_mask, 'TimeAlive']
    
    print(f"   📈 Datos de regresión: {len(y_regression)} muestras")
    
    if len(y_regression) < 100:
        print(f"   ⚠️ Muy pocas muestras para regresión ({len(y_regression)})")
        print(f"   🔄 Intentando usar variable alternativa...")
        
        # Buscar variables alternativas para regresión
        alternate_targets = ['RoundKills', 'MatchKills', 'RoundStartingEquipmentValue']
        
        for alt_target in alternate_targets:
            if alt_target in df_model.columns:
                alt_values = df_model[alt_target].dropna()
                if len(alt_values) > 1000:  # Suficientes datos
                    y_regression = alt_values
                    survival_mask = df_model[alt_target].notna()
                    print(f"   ✅ Usando {alt_target} como variable objetivo de regresión")
                    break
    
    if len(y_regression) > 0:
        print(f"   📊 Variable regresión - Min: {y_regression.min():.2f}, Max: {y_regression.max():.2f}, Media: {y_regression.mean():.2f}")
    else:
        print(f"   ❌ No hay datos suficientes para regresión")
    
    # 7. Codificación de variables categóricas
    print(f"\n🏷️ CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    
    # Codificar Team
    if 'Team' in df_model.columns:
        unique_teams = df_model['Team'].dropna().unique()
        team_mapping = {team: i for i, team in enumerate(unique_teams)}
        df_model['Team_Encoded'] = df_model['Team'].map(team_mapping).fillna(0)
        available_features.append('Team_Encoded')
        print(f"   ✅ Team codificado: {team_mapping}")
    
    # Codificar MatchWinner (maneja NaN)
    if 'MatchWinner' in df_model.columns:
        # Rellenar NaN antes de convertir a entero
        df_model['MatchWinner_Encoded'] = df_model['MatchWinner'].fillna(False).astype(int)
        available_features.append('MatchWinner_Encoded')
        nan_count = df_model['MatchWinner'].isnull().sum()
        print(f"   ✅ MatchWinner codificado ({nan_count} NaN convertidos a 0)")
        
        # Mostrar distribución final
        distribution = df_model['MatchWinner_Encoded'].value_counts().to_dict()
        print(f"   📊 Distribución final MatchWinner_Encoded: {distribution}")
    
    # 8. Preparación final de datos
    print(f"\n🔄 PREPARACIÓN FINAL DE DATOS")
    
    # Seleccionar características
    X = df_model[available_features].copy()
    
    # Eliminar filas donde la variable objetivo es nula
    valid_classification_mask = y_classification.notna()
    X_classification = X[valid_classification_mask]
    y_classification_clean = y_classification[valid_classification_mask]
    
    # Inicializar variable de control para regresión
    has_regression_data = False
    
    # Para regresión, usar solo datos con variable objetivo válida
    if len(y_regression) > 100:  # Solo si hay suficientes datos
        X_regression = X[survival_mask]
        y_regression_clean = y_regression
        
        print(f"   📊 Datos clasificación finales: {X_classification.shape}")
        print(f"   📊 Datos regresión finales: {X_regression.shape}")
        
        has_regression_data = True
    else:
        print(f"   ⚠️ Datos insuficientes para regresión ({len(y_regression)} muestras)")
        print(f"   🔄 Solo se entrenará modelo de clasificación")
        
        # Crear datos vacíos para regresión (para mantener compatibilidad)
        X_regression = X_classification[:100]  # Datos dummy
        y_regression_clean = pd.Series(range(100))  # Datos dummy
        has_regression_data = False
    
    # 9. Imputación de valores faltantes
    print(f"\n🔧 IMPUTACIÓN DE VALORES FALTANTES")
    
    # Crear imputadores
    imputer_classification = SimpleImputer(strategy='median')
    imputer_regression = SimpleImputer(strategy='median')
    
    # Imputar datos de clasificación
    print(f"   📊 NaN en datos de clasificación antes: {X_classification.isnull().sum().sum()}")
    X_classification_imputed = imputer_classification.fit_transform(X_classification)
    print(f"   ✅ Imputación completada para clasificación")
    
    # Imputar datos de regresión (solo si hay datos válidos)
    if has_regression_data and len(X_regression) > 0:
        print(f"   📊 NaN en datos de regresión antes: {X_regression.isnull().sum().sum()}")
        X_regression_imputed = imputer_regression.fit_transform(X_regression)
        print(f"   ✅ Imputación completada para regresión")
        
        # Verificar que no queden NaN en regresión
        assert not np.isnan(X_regression_imputed).any(), "Todavía hay NaN en datos de regresión"
    else:
        # Crear datos dummy para mantener compatibilidad
        print(f"   ⚠️ Sin datos de regresión válidos, creando datos dummy")
        X_regression_imputed = X_classification_imputed[:100]  # Datos dummy
        imputer_regression = imputer_classification  # Usar el mismo imputador
    
    # Verificar que no queden NaN
    assert not np.isnan(X_classification_imputed).any(), "Todavía hay NaN en datos de clasificación"
    print(f"   ✅ Verificación: No quedan valores NaN en clasificación")
    
    # 10. División de datos
    print(f"\n📊 DIVISIÓN DE DATOS")
    
    # Clasificación
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_classification_imputed, y_classification_clean, 
        test_size=0.2, random_state=42, 
        stratify=y_classification_clean if len(np.unique(y_classification_clean)) > 1 else None
    )
    
    # Regresión
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression_imputed, y_regression_clean, 
        test_size=0.2, random_state=42
    )
    
    print(f"   ✅ Entrenamiento clasificación: {X_train_clf.shape}")
    print(f"   ✅ Prueba clasificación: {X_test_clf.shape}")
    print(f"   ✅ Entrenamiento regresión: {X_train_reg.shape}")
    print(f"   ✅ Prueba regresión: {X_test_reg.shape}")
    
    # 11. Escalado de datos
    print(f"\n⚖️ ESCALADO DE DATOS")
    
    scaler_clf = StandardScaler()
    scaler_reg = StandardScaler()
    
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)
    
    # Escalado para regresión (solo si hay datos válidos)
    if has_regression_data:
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    else:
        # Usar los mismos datos de clasificación como dummy
        X_train_reg_scaled = X_train_clf_scaled[:50]
        X_test_reg_scaled = X_test_clf_scaled[:20]
        scaler_reg = scaler_clf  # Usar el mismo scaler
    
    print(f"   ✅ Escalado completado")
    
    # Verificación final de NaN
    datasets_to_check = [("X_train_clf_scaled", X_train_clf_scaled)]
    if has_regression_data:
        datasets_to_check.append(("X_train_reg_scaled", X_train_reg_scaled))
    
    for name, data in datasets_to_check:
        if np.isnan(data).any():
            print(f"❌ ERROR: {name} contiene NaN después del escalado")
            return None
        else:
            print(f"   ✅ {name}: Sin valores NaN")
    
    # 12. Entrenamiento de modelos
    print(f"\n🤖 ENTRENAMIENTO DE MODELOS")
    
    # Modelos de clasificación (solo Random Forest por robustez)
    print(f"   🔵 Entrenando Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_classifier.fit(X_train_clf_scaled, y_train_clf)
    rf_clf_score = rf_classifier.score(X_test_clf_scaled, y_test_clf)
    print(f"      ✅ Random Forest Classifier - Accuracy: {rf_clf_score:.4f}")
    
    # Intentar Logistic Regression con configuración robusta
    print(f"   🔵 Entrenando Logistic Regression...")
    try:
        logistic_classifier = LogisticRegression(
            random_state=42, 
            max_iter=2000,
            solver='liblinear',  # Más robusto para datasets problemáticos
            C=1.0
        )
        logistic_classifier.fit(X_train_clf_scaled, y_train_clf)
        log_clf_score = logistic_classifier.score(X_test_clf_scaled, y_test_clf)
        print(f"      ✅ Logistic Regression - Accuracy: {log_clf_score:.4f}")
    except Exception as e:
        print(f"      ⚠️ Logistic Regression falló: {e}")
        print(f"      🔄 Usando solo Random Forest para clasificación")
        logistic_classifier = rf_classifier  # Fallback
        log_clf_score = rf_clf_score
    
    # Modelos de regresión (solo si hay datos suficientes)
    if has_regression_data:
        print(f"   🔴 Entrenando Random Forest Regressor...")
        rf_regressor = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_regressor.fit(X_train_reg_scaled, y_train_reg)
        rf_reg_score = rf_regressor.score(X_test_reg_scaled, y_test_reg)
        print(f"      ✅ Random Forest Regressor - R²: {rf_reg_score:.4f}")
        
        print(f"   🔴 Entrenando Linear Regression...")
        linear_regressor = LinearRegression()
        linear_regressor.fit(X_train_reg_scaled, y_train_reg)
        linear_reg_score = linear_regressor.score(X_test_reg_scaled, y_test_reg)
        print(f"      ✅ Linear Regression - R²: {linear_reg_score:.4f}")
    else:
        print(f"   ⚠️ Saltando entrenamiento de regresión (datos insuficientes)")
        # Crear modelos dummy que devuelvan valores predeterminados
        rf_regressor = type('DummyRegressor', (), {
            'predict': lambda self, X: np.full(len(X), 50.0),
            'score': lambda self, X, y: 0.0
        })()
        linear_regressor = rf_regressor
        rf_reg_score = 0.0
        linear_reg_score = 0.0
    
    # 13. Guardar modelos
    print(f"\n💾 GUARDANDO MODELOS")
    
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Crear mapeo para la API
    team_mapping = {}
    if 'Team' in df_model.columns:
        unique_teams = df_model['Team'].dropna().unique()
        team_mapping = {team: i for i, team in enumerate(unique_teams)}
    else:
        team_mapping = {'Terrorist': 0, 'Counter-Terrorist': 1}
    
    model_package = {
        'models': {
            'rf_classifier': rf_classifier,
            'logistic_classifier': logistic_classifier,
            'rf_regressor': rf_regressor,
            'linear_regressor': linear_regressor
        },
        'scalers': {
            'scaler_classification': scaler_clf,
            'scaler_regression': scaler_reg
        },
        'imputers': {
            'imputer_classification': imputer_classification,
            'imputer_regression': imputer_regression
        },
        'features': {
            'available_features': available_features,
            'categorical_mapping': {
                'Team': team_mapping,
                'MatchWinner': {False: 0, True: 1}
            }
        },
        'metadata': {
            'training_samples_clf': len(X_train_clf),
            'training_samples_reg': len(X_train_reg) if has_regression_data else 0,
            'feature_count': len(available_features),
            'rf_clf_accuracy': rf_clf_score,
            'log_clf_accuracy': log_clf_score,
            'rf_reg_r2': rf_reg_score,
            'linear_reg_r2': linear_reg_score,
            'data_source': file_path,
            'team_mapping': team_mapping,
            'has_valid_regression': has_regression_data
        }
    }
    
    # Guardar paquete completo
    joblib.dump(model_package, f"{models_dir}/csgo_model_package.pkl")
    print(f"   ✅ Paquete principal guardado: csgo_model_package.pkl")
    
    # Guardar información adicional
    with open(f"{models_dir}/features.txt", 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")
    
    import json
    config = {
        "model_version": "1.0.0",
        "training_date": pd.Timestamp.now().isoformat(),
        "features_count": len(available_features),
        "models_performance": {
            "rf_classifier_accuracy": rf_clf_score,
            "logistic_classifier_accuracy": log_clf_score,
            "rf_regressor_r2": rf_reg_score if has_regression_data else 0.0,
            "linear_regressor_r2": linear_reg_score if has_regression_data else 0.0
        },
        "data_info": {
            "total_samples": len(df),
            "classification_samples": len(X_classification),
            "regression_samples": len(X_regression) if has_regression_data else 0,
            "features": available_features,
            "has_valid_regression": has_regression_data
        }
    }
    
    with open(f"{models_dir}/model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ Configuración guardada: model_config.json")
    print(f"   ✅ Lista de características: features.txt")
    
    # 14. Resumen final
    print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print(f"=" * 50)
    print(f"📊 RESUMEN FINAL:")
    print(f"   🎯 Características utilizadas: {len(available_features)}")
    print(f"   📈 Muestras clasificación: {len(X_train_clf):,}")
    if has_regression_data:
        print(f"   📈 Muestras regresión: {len(X_train_reg):,}")
        print(f"   🏆 Mejor regresor: {'Random Forest' if rf_reg_score > linear_reg_score else 'Linear'} ({max(rf_reg_score, linear_reg_score):.3f} R²)")
    else:
        print(f"   ⚠️ Sin datos suficientes para regresión")
    print(f"   🏆 Mejor clasificador: Random Forest ({rf_clf_score:.1%})")
    print(f"   💾 Archivos guardados en: {models_dir}/")
    
    print(f"\n🚀 PRÓXIMO PASO:")
    print(f"   python app.py")
    
    return model_package

if __name__ == "__main__":
    entrenar_y_guardar_modelos()