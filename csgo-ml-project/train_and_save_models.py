#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar y guardar modelos de clasificación y regresión
para el proyecto CSGO ML
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

def cargar_y_preparar_datos():
    """Carga y prepara los datos para el modelado"""
    print("📊 Cargando datos...")
    
    # Cargar datos
    file_path = "../data/Anexo ET_demo_round_traces_2022.csv"
    df = pd.read_csv(file_path, delimiter=';')
    
    # Limpieza básica
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Convertir columnas numéricas
    cols_a_convertir = ['TimeAlive', 'TravelledDistance', 'FirstKillTime']
    for col in cols_a_convertir:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con valores críticos faltantes
    df.dropna(subset=['Team', 'MatchWinner'], inplace=True)
    
    # Imputación de valores nulos
    for col in ['TimeAlive', 'TravelledDistance', 'FirstKillTime']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    # Convertir columnas booleanas
    bool_cols = ['Survived', 'AbnormalMatch']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Convertir columnas enteras
    int_cols = ['InternalTeamId', 'MatchId', 'RoundId', 'RLethalGrenadesThrown',
                'RNonLethalGrenadesThrown', 'PrimaryPistol', 'RoundKills', 'RoundAssists',
                'RoundHeadshots', 'RoundFlankKills', 'RoundStartingEquipmentValue',
                'TeamStartingEquipmentValue', 'MatchKills', 'MatchFlankKills',
                'MatchAssists', 'MatchHeadshots']
    
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Codificación de variables categóricas
    df = pd.get_dummies(df, columns=['Map', 'RoundWinner'], drop_first=True)
    
    print(f"✅ Datos preparados: {df.shape}")
    return df

def preparar_datos_modelado(df):
    """Prepara los datos para clasificación y regresión"""
    print("🔧 Preparando datos para modelado...")
    
    # Características para el modelo
    features = [
        'MatchKills', 'MatchAssists', 'MatchHeadshots',
        'PrimaryAssaultRifle', 'PrimarySniperRifle',
        'TravelledDistance', 'RoundStartingEquipmentValue',
        'RoundKills', 'RoundAssists', 'RoundHeadshots',
        'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown',
        'RoundFlankKills', 'TeamStartingEquipmentValue',
        'MatchFlankKills', 'PrimaryPistol'
    ]
    
    # Filtrar características que existen en el dataset
    available_features = [f for f in features if f in df.columns]
    print(f"📋 Características disponibles: {len(available_features)}/{len(features)}")
    
    # Datos para clasificación (sobrevive o no)
    df_clf = df.copy()
    df_clf['Survived'] = (df_clf['TimeAlive'] > 0).astype(int)
    
    # Datos para regresión (solo valores positivos de TimeAlive)
    df_reg = df[df['TimeAlive'] > 0].copy()
    
    print(f"🎯 Datos de clasificación: {df_clf.shape}")
    print(f"📊 Datos de regresión: {df_reg.shape}")
    
    return df_clf, df_reg, available_features

def entrenar_modelos(df_clf, df_reg, features):
    """Entrena los modelos de clasificación y regresión"""
    print("🤖 Entrenando modelos...")
    
    # Preparar datos de clasificación
    X_clf = df_clf[features]
    y_clf = df_clf['Survived']
    
    # Preparar datos de regresión
    X_reg = df_reg[features]
    y_reg = df_reg['TimeAlive']
    
    # División de datos
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Imputación y escalado
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Aplicar a datos de clasificación
    X_clf_train_imputed = imputer.fit_transform(X_clf_train)
    X_clf_test_imputed = imputer.transform(X_clf_test)
    X_clf_train_scaled = scaler.fit_transform(X_clf_train_imputed)
    X_clf_test_scaled = scaler.transform(X_clf_test_imputed)
    
    # Aplicar a datos de regresión
    X_reg_train_imputed = imputer.transform(X_reg_train)
    X_reg_test_imputed = imputer.transform(X_reg_test)
    X_reg_train_scaled = scaler.transform(X_reg_train_imputed)
    X_reg_test_scaled = scaler.transform(X_reg_test_imputed)
    
    # Entrenar modelos de clasificación
    print("🔵 Entrenando clasificadores...")
    
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_clf_train_scaled, y_clf_train)
    rf_clf_pred = rf_classifier.predict(X_clf_test_scaled)
    rf_clf_accuracy = accuracy_score(y_clf_test, rf_clf_pred)
    print(f"   ✅ Random Forest Classifier - Accuracy: {rf_clf_accuracy:.4f}")
    
    # Logistic Regression
    logistic_classifier = LogisticRegression(random_state=42, max_iter=1000)
    logistic_classifier.fit(X_clf_train_scaled, y_clf_train)
    log_clf_pred = logistic_classifier.predict(X_clf_test_scaled)
    log_clf_accuracy = accuracy_score(y_clf_test, log_clf_pred)
    print(f"   ✅ Logistic Regression - Accuracy: {log_clf_accuracy:.4f}")
    
    # Entrenar modelos de regresión
    print("🔴 Entrenando regresores...")
    
    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_reg_train_scaled, y_reg_train)
    rf_reg_pred = rf_regressor.predict(X_reg_test_scaled)
    rf_reg_r2 = r2_score(y_reg_test, rf_reg_pred)
    print(f"   ✅ Random Forest Regressor - R²: {rf_reg_r2:.4f}")
    
    # Linear Regression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_reg_train_scaled, y_reg_train)
    lin_reg_pred = linear_regressor.predict(X_reg_test_scaled)
    lin_reg_r2 = r2_score(y_reg_test, lin_reg_pred)
    print(f"   ✅ Linear Regression - R²: {lin_reg_r2:.4f}")
    
    return {
        'rf_classifier': rf_classifier,
        'logistic_classifier': logistic_classifier,
        'rf_regressor': rf_regressor,
        'linear_regressor': linear_regressor,
        'imputer': imputer,
        'scaler': scaler,
        'features': features,
        'metrics': {
            'rf_clf_accuracy': rf_clf_accuracy,
            'log_clf_accuracy': log_clf_accuracy,
            'rf_reg_r2': rf_reg_r2,
            'lin_reg_r2': lin_reg_r2
        }
    }

def guardar_modelos(modelos, output_dir='saved_models'):
    """Guarda los modelos entrenados"""
    print("💾 Guardando modelos...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelos individuales
    joblib.dump(modelos['rf_classifier'], f'{output_dir}/rf_classifier.pkl')
    joblib.dump(modelos['logistic_classifier'], f'{output_dir}/logistic_classifier.pkl')
    joblib.dump(modelos['rf_regressor'], f'{output_dir}/rf_regressor.pkl')
    joblib.dump(modelos['linear_regressor'], f'{output_dir}/linear_regressor.pkl')
    
    # Guardar preprocesadores
    joblib.dump(modelos['imputer'], f'{output_dir}/imputer.pkl')
    joblib.dump(modelos['scaler'], f'{output_dir}/scaler.pkl')
    
    # Guardar información de características
    joblib.dump(modelos['features'], f'{output_dir}/features.pkl')
    
    # Guardar métricas
    joblib.dump(modelos['metrics'], f'{output_dir}/metrics.pkl')
    
    # Guardar diccionario completo de modelos
    joblib.dump(modelos, f'{output_dir}/modelos_completos.pkl')
    
    print(f"✅ Modelos guardados en: {output_dir}/")
    print(f"📊 Métricas guardadas:")
    for key, value in modelos['metrics'].items():
        print(f"   {key}: {value:.4f}")

def main():
    """Función principal"""
    print("🚀 Iniciando entrenamiento y guardado de modelos...")
    
    # Cargar y preparar datos
    df = cargar_y_preparar_datos()
    
    # Preparar datos para modelado
    df_clf, df_reg, features = preparar_datos_modelado(df)
    
    # Entrenar modelos
    modelos = entrenar_modelos(df_clf, df_reg, features)
    
    # Guardar modelos
    guardar_modelos(modelos)
    
    print("🎉 ¡Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main() 