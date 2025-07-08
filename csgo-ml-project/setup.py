# setup.py - Script para configurar todo automáticamente
import os
import sys
import subprocess
import time

def print_header(title):
    """Imprimir encabezado formateado"""
    print("\n" + "="*60)
    print(f"🎮 {title}")
    print("="*60)

def run_command(command, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completado")
            return True
        else:
            print(f"❌ Error en {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        return False

def check_file_exists(filepath, description):
    """Verificar que un archivo existe"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ No encontrado {description}: {filepath}")
        return False

def setup_environment():
    """Configurar el entorno completo"""
    print_header("CONFIGURACIÓN AUTOMÁTICA DEL PROYECTO CS:GO ML")
    
    # 1. Verificar Python
    print("\n🐍 Verificando Python...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor} detectado")
    else:
        print(f"❌ Se requiere Python 3.8+. Tienes {python_version.major}.{python_version.minor}")
        return False
    
    # 2. Verificar estructura de archivos
    print("\n📁 Verificando estructura de archivos...")
    
    required_files = [
        ("../data/Anexo ET_demo_round_traces_2022.csv", "Dataset CSV"),
        ("save_model.py", "Script de entrenamiento"),
        ("app.py", "API Flask"),
        ("requirements.txt", "Archivo de dependencias")
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Faltan archivos requeridos. Verifica la estructura del proyecto.")
        return False
    
    # 3. Instalar dependencias
    print("\n📦 Instalando dependencias...")
    if not run_command("pip install -r requirements.txt", "Instalación de dependencias"):
        print("💡 Intenta: pip install --upgrade pip")
        print("💡 O usa: python -m pip install -r requirements.txt")
        return False
    
    # 4. Entrenar modelos
    print("\n🤖 Entrenando modelos de Machine Learning...")
    print("⏱️ Esto puede tomar unos minutos...")
    
    if not run_command("python save_model.py", "Entrenamiento de modelos"):
        print("❌ Error entrenando modelos. Verifica el dataset.")
        return False
    
    # 5. Verificar modelos guardados
    print("\n💾 Verificando modelos guardados...")
    
    model_files = [
        ("saved_models/csgo_model_package.pkl", "Paquete principal de modelos"),
        ("saved_models/model_config.json", "Configuración del modelo"),
        ("saved_models/features.txt", "Lista de características")
    ]
    
    for filepath, description in model_files:
        if not check_file_exists(filepath, description):
            print(f"⚠️  Advertencia: {description} no encontrado")
    
    # 6. Prueba rápida de la API
    print("\n🧪 Realizando prueba rápida...")
    
    # Importar y probar carga de modelos
    try:
        print("   Importando Flask...")
        import flask
        print("   ✅ Flask disponible")
        
        print("   Probando carga de modelos...")
        import joblib
        if os.path.exists("saved_models/csgo_model_package.pkl"):
            model_package = joblib.load("saved_models/csgo_model_package.pkl")
            print("   ✅ Modelos cargan correctamente")
            
            # Mostrar información del modelo
            metadata = model_package.get('metadata', {})
            print(f"   📊 Accuracy Clasificador: {metadata.get('rf_clf_accuracy', 'N/A')}")
            print(f"   📈 R² Regresor: {metadata.get('rf_reg_r2', 'N/A')}")
            print(f"   🔢 Características: {metadata.get('feature_count', 'N/A')}")
            
        else:
            print("   ❌ No se encontró el archivo de modelos")
            return False
            
    except ImportError as e:
        print(f"   ❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error probando modelos: {e}")
        return False
    
    return True

def start_api():
    """Iniciar la API"""
    print_header("INICIANDO API")
    
    print("\n🚀 La API se iniciará en unos segundos...")
    print("📍 URL: http://localhost:5000")
    print("💡 Presiona Ctrl+C para detener la API")
    print("\n⏱️ Esperando 3 segundos...")
    
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🎯 ¡Iniciando servidor!")
    print("-" * 40)
    
    # Importar y ejecutar la API
    try:
        import app
        # La API se ejecutará automáticamente si load_models() es exitoso
    except KeyboardInterrupt:
        print("\n\n⏹️  API detenida por el usuario")
    except Exception as e:
        print(f"\n❌ Error iniciando API: {e}")

def show_usage_examples():
    """Mostrar ejemplos de uso"""
    print_header("EJEMPLOS DE USO")
    
    examples = [
        {
            "title": "Verificar que la API funciona",
            "command": "curl http://localhost:5000/health"
        },
        {
            "title": "Predicción simple",
            "command": '''curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"Team": "Terrorist", "RoundKills": 2}'
'''
        },
        {
            "title": "Ejecutar todas las pruebas",
            "command": "python test_api.py"
        }
    ]
    
    for example in examples:
        print(f"\n🔹 {example['title']}:")
        print(f"   {example['command']}")
    
    print(f"\n📖 Más información en README.md")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configurar proyecto CS:GO ML")
    parser.add_argument("--setup-only", action="store_true", 
                       help="Solo configurar, no iniciar API")
    parser.add_argument("--start-only", action="store_true", 
                       help="Solo iniciar API (asumir ya configurado)")
    parser.add_argument("--examples", action="store_true", 
                       help="Mostrar ejemplos de uso")
    
    args = parser.parse_args()
    
    if args.examples:
        show_usage_examples()
        return
    
    if args.start_only:
        start_api()
        return
    
    # Configuración completa
    success = setup_environment()
    
    if success:
        print_header("¡CONFIGURACIÓN EXITOSA!")
        print("\n✅ Todo está listo para usar")
        print("🎯 El proyecto CS:GO ML está configurado correctamente")
        
        if not args.setup_only:
            print("\n💡 ¿Quieres iniciar la API ahora? (y/n): ", end="")
            response = input().lower().strip()
            
            if response in ['y', 'yes', 'sí', 's']:
                start_api()
            else:
                print("\n🚀 Para iniciar la API más tarde, ejecuta:")
                print("   python app.py")
                print("\n🧪 Para probar la API, ejecuta:")
                print("   python test_api.py")
        else:
            print("\n🚀 Para iniciar la API:")
            print("   python app.py")
    else:
        print_header("❌ CONFIGURACIÓN FALLIDA")
        print("\n💡 Pasos para solucionar:")
        print("1. Verifica que tienes Python 3.8+")
        print("2. Instala las dependencias: pip install -r requirements.txt")
        print("3. Asegúrate de que el archivo CSV está en ../data/")
        print("4. Ejecuta: python save_model.py")
        print("5. Luego: python app.py")

if __name__ == "__main__":
    main()