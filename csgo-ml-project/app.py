# app.py - Aplicación Streamlit para CS:GO ML Project
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import joblib
import io

# Configuración de la página
st.set_page_config(
    page_title="CS:GO ML Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def load_and_prepare_data():
    file_path = "../data/Anexo ET_demo_round_traces_2022.csv"
    df = pd.read_csv(file_path, delimiter=';')
    return df

@st.cache_resource
def train_models(df):
    """Entrena los modelos"""
    features = ['MatchKills', 'MatchAssists', 'MatchHeadshots', 
                'PrimaryAssaultRifle', 'PrimarySniperRifle', 
                'TravelledDistance', 'RoundStartingEquipmentValue']
    
    X = df[features]
    y_class = df['Survived']
    y_reg = df['TimeAlive']
    
    # División de datos
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos de clasificación
    models_class = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Modelos de regresión
    models_reg = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR()
    }
    
    # Entrenar modelos
    trained_models = {'classification': {}, 'regression': {}}
    results = {'classification': {}, 'regression': {}}
    
    for name, model in models_class.items():
        model.fit(X_train_scaled, y_class_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_class_test, y_pred)
        trained_models['classification'][name] = model
        results['classification'][name] = {'accuracy': acc, 'predictions': y_pred}
    
    for name, model in models_reg.items():
        model.fit(X_train_scaled, y_reg_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_reg_test, y_pred)
        mae = mean_absolute_error(y_reg_test, y_pred)
        trained_models['regression'][name] = model
        results['regression'][name] = {'r2': r2, 'mae': mae, 'predictions': y_pred}
    
    return trained_models, scaler, results, features, X_test, y_class_test, y_reg_test

def create_prediction_interface(models, scaler, features):
    """Interface para hacer predicciones"""
    st.markdown("### 🔮 Hacer Predicciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Estadísticas del Jugador")
        match_kills = st.slider("Match Kills", 0, 30, 10)
        match_assists = st.slider("Match Assists", 0, 20, 5)
        match_headshots = st.slider("Match Headshots", 0, 15, 3)
        travelled_distance = st.slider("Distancia Recorrida", 1000, 8000, 3000)
    
    with col2:
        st.markdown("#### Equipamiento y Armas")
        primary_assault = st.slider("% Uso Rifle de Asalto", 0.0, 1.0, 0.6)
        primary_sniper = st.slider("% Uso Rifle de Francotirador", 0.0, 1.0, 0.1)
        equipment_value = st.slider("Valor del Equipamiento", 1000, 15000, 5000)
    
    # Preparar datos para predicción
    input_data = np.array([[
        match_kills, match_assists, match_headshots,
        primary_assault, primary_sniper,
        travelled_distance, equipment_value
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    if st.button("🚀 Realizar Predicción", type="primary"):
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Predicción de Supervivencia")
            for name, model in models['classification'].items():
                prob = model.predict_proba(input_scaled)[0]
                survival_prob = prob[1] * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{name}</h4>
                    <p><strong>Probabilidad de Supervivencia: {survival_prob:.1f}%</strong></p>
                    <p>Predicción: {'✅ Sobrevive' if survival_prob > 50 else '❌ No sobrevive'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⏱️ Predicción de Tiempo de Vida")
            for name, model in models['regression'].items():
                time_pred = model.predict(input_scaled)[0]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{name}</h4>
                    <p><strong>Tiempo Predicho: {time_pred:.1f} segundos</strong></p>
                    <p>Rendimiento: {'🔥 Excelente' if time_pred > 60 else '👍 Bueno' if time_pred > 30 else '⚠️ Mejorable'}</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🎮 CS:GO Machine Learning Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Análisis Predictivo de Rendimiento en Counter-Strike: Global Offensive")
    
    # Cargar datos y entrenar modelos
    with st.spinner("Cargando datos y entrenando modelos..."):
        df = load_and_prepare_data()
        models, scaler, results, features, X_test, y_class_test, y_reg_test = train_models(df)
    
    # Sidebar para navegación
    st.sidebar.markdown("## 📊 Navegación")
    tab_selection = st.sidebar.radio(
        "Selecciona una sección:",
        ["🏠 Dashboard", "📈 Análisis de Datos", "🤖 Modelos ML", "🔮 Predicciones", "📋 Información"]
    )
    
    if tab_selection == "🏠 Dashboard":
        dashboard_tab(df, results)
    elif tab_selection == "📈 Análisis de Datos":
        data_analysis_tab(df)
    elif tab_selection == "🤖 Modelos ML":
        models_tab(results)
    elif tab_selection == "🔮 Predicciones":
        create_prediction_interface(models, scaler, features)
    elif tab_selection == "📋 Información":
        info_tab()

def dashboard_tab(df, results):
    """Dashboard principal con métricas clave"""
    st.markdown("## 🏠 Dashboard Principal")
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Partidas", f"{len(df):,}")
    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Tasa de Supervivencia", f"{survival_rate:.1f}%")
    with col3:
        avg_time = df['TimeAlive'].mean()
        st.metric("Tiempo Promedio de Vida", f"{avg_time:.1f}s")
    with col4:
        avg_kills = df['MatchKills'].mean()
        st.metric("Kills Promedio", f"{avg_kills:.1f}")
    
    st.markdown("---")
    
    # Gráficos del dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de supervivencia por mapa
        fig_map = px.bar(
            df.groupby('Map')['Survived'].mean().reset_index(),
            x='Map', y='Survived',
            title="📍 Tasa de Supervivencia por Mapa",
            color='Survived',
            color_continuous_scale='Viridis'
        )
        fig_map.update_layout(showlegend=False)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        # Distribución de tiempo de vida
        fig_time = px.histogram(
            df, x='TimeAlive',
            title="⏱️ Distribución del Tiempo de Vida",
            nbins=30,
            color_discrete_sequence=['#FF6B35']
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Rendimiento de modelos
    st.markdown("### 🏆 Rendimiento de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Clasificación (Supervivencia)")
        class_results = []
        for name, result in results['classification'].items():
            class_results.append({'Modelo': name, 'Accuracy': result['accuracy']})
        
        class_df = pd.DataFrame(class_results)
        fig_class = px.bar(
            class_df, x='Modelo', y='Accuracy',
            title="Accuracy de Modelos de Clasificación",
            color='Accuracy',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col2:
        st.markdown("#### Regresión (Tiempo de Vida)")
        reg_results = []
        for name, result in results['regression'].items():
            reg_results.append({'Modelo': name, 'R²': result['r2']})
        
        reg_df = pd.DataFrame(reg_results)
        fig_reg = px.bar(
            reg_df, x='Modelo', y='R²',
            title="R² de Modelos de Regresión",
            color='R²',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_reg, use_container_width=True)

def data_analysis_tab(df):
    """Tab de análisis de datos"""
    st.markdown("## 📈 Análisis Exploratorio de Datos")
    
    # Mostrar datos
    with st.expander("👀 Ver Datos Crudos"):
        st.dataframe(df.head(100), use_container_width=True)
    
    # Estadísticas descriptivas
    st.markdown("### 📊 Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Matriz de correlación
    st.markdown("### 🔗 Matriz de Correlación")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlación de Variables Numéricas"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análisis por equipos
    st.markdown("### 👥 Análisis por Equipos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_stats = df.groupby('Team').agg({
            'MatchKills': 'mean',
            'MatchAssists': 'mean',
            'TimeAlive': 'mean',
            'Survived': 'mean'
        }).round(2)
        st.dataframe(team_stats, use_container_width=True)
    
    with col2:
        fig_team = px.box(
            df, x='Team', y='TimeAlive',
            title="Distribución de Tiempo de Vida por Equipo"
        )
        st.plotly_chart(fig_team, use_container_width=True)

def models_tab(results):
    """Tab de información de modelos"""
    st.markdown("## 🤖 Información de Modelos")
    
    # Clasificación
    st.markdown("### 🎯 Modelos de Clasificación")
    
    class_data = []
    for name, result in results['classification'].items():
        class_data.append({
            'Modelo': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Descripción': get_model_description(name, 'classification')
        })
    
    class_df = pd.DataFrame(class_data)
    st.dataframe(class_df, use_container_width=True)
    
    # Regresión
    st.markdown("### 📊 Modelos de Regresión")
    
    reg_data = []
    for name, result in results['regression'].items():
        reg_data.append({
            'Modelo': name,
            'R²': f"{result['r2']:.4f}",
            'MAE': f"{result['mae']:.2f}",
            'Descripción': get_model_description(name, 'regression')
        })
    
    reg_df = pd.DataFrame(reg_data)
    st.dataframe(reg_df, use_container_width=True)
    
    # Interpretación de características
    st.markdown("### 🔍 Características del Modelo")
    
    features_info = {
        'MatchKills': 'Número total de eliminaciones en la partida',
        'MatchAssists': 'Número total de asistencias en la partida',
        'MatchHeadshots': 'Número de eliminaciones con headshot',
        'PrimaryAssaultRifle': 'Porcentaje de uso de rifles de asalto',
        'PrimarySniperRifle': 'Porcentaje de uso de rifles de francotirador',
        'TravelledDistance': 'Distancia total recorrida en el mapa',
        'RoundStartingEquipmentValue': 'Valor del equipamiento inicial'
    }
    
    for feature, description in features_info.items():
        st.markdown(f"- **{feature}**: {description}")

def get_model_description(name, model_type):
    """Obtiene descripción del modelo"""
    descriptions = {
        'Random Forest': {
            'classification': 'Ensemble de árboles de decisión para clasificación robusta',
            'regression': 'Ensemble de árboles de decisión para predicción continua'
        },
        'Logistic Regression': {
            'classification': 'Modelo lineal para clasificación binaria/multiclase'
        },
        'Linear Regression': {
            'regression': 'Modelo lineal para predicción de valores continuos'
        },
        'SVM': {
            'classification': 'Máquinas de vectores de soporte para clasificación'
        },
        'SVR': {
            'regression': 'Máquinas de vectores de soporte para regresión'
        }
    }
    
    return descriptions.get(name, {}).get(model_type, 'Descripción no disponible')

def info_tab():
    """Tab de información del proyecto"""
    st.markdown("## 📋 Información del Proyecto")
    
    st.markdown("""
    ### 🎮 CS:GO Machine Learning Analyzer
    
    Esta aplicación utiliza técnicas de Machine Learning para analizar y predecir el rendimiento 
    de jugadores en Counter-Strike: Global Offensive.
    
    #### 🎯 Objetivos del Proyecto
    - **Clasificación**: Predecir si un jugador sobrevivirá una ronda
    - **Regresión**: Estimar el tiempo de supervivencia del jugador
    - **Análisis**: Identificar factores clave que influyen en el rendimiento
    
    #### 📊 Características Analizadas
    - Estadísticas de combate (kills, assists, headshots)
    - Uso de armas primarias
    - Movimiento en el mapa
    - Valor del equipamiento
    
    #### 🤖 Modelos Implementados
    
    **Clasificación:**
    - Random Forest Classifier
    - Logistic Regression
    - Support Vector Machine (SVM)
    
    **Regresión:**
    - Random Forest Regressor
    - Linear Regression
    - Support Vector Regression (SVR)
    
    #### 🛠️ Tecnologías Utilizadas
    - **Python**: Lenguaje de programación principal
    - **Streamlit**: Framework para la aplicación web
    - **Scikit-learn**: Biblioteca de machine learning
    - **Plotly**: Visualizaciones interactivas
    - **Pandas**: Manipulación de datos
    
    #### 👥 Equipo de Desarrollo
    - Cristian Olivares
    - Dario García
    - Jean Paul Leyton
    
    #### 📧 Contacto
    Para más información sobre este proyecto, contacta al equipo de desarrollo.
    """)
    
    # Métricas del sistema
    st.markdown("### 📈 Métricas del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelos Entrenados", "6")
    with col2:
        st.metric("Características", "7")
    with col3:
        st.metric("Versión", "1.0.0")

if __name__ == "__main__":
    main()