import pandas as pd

# Importamos el archivo csv
file_path = "../data/Anexo ET_demo_round_traces_2022.csv"
# Leemos el CSV en un DataFrame
df = pd.read_csv(file_path, delimiter=';')

# Dimensiones del dataset
print("Número de filas y columnas:", df.shape)

