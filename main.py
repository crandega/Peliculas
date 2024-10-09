import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sentence_transformers import SentenceTransformer, util

def info_rele(df):
    """
    Crea una nueva columna en el DataFrame que combina el nombre del director y el valor ganado
    de la película para proporcionar más contexto.
    """
    # Asegúrate de que las columnas 'Cast' y 'Info' existan en el DataFrame antes de usarlas
    df['informacion_relevante'] = df.apply(lambda row: f"Director: {row['Cast']}, Valor ganado: {row['Info']}", axis=1)
    return df

def main():
    # Cargar el dataset
    df = pd.read_csv('src\IMDB top 1000.csv')
    # print(df.head())  # Mostrar las primeras filas del DataFrame

    # Crear una nueva columna con información relevante (sin cambiar la búsqueda)
    df = info_rele(df)
    # print(df[['Title', 'informacion_relevante']].head())  # Mostrar algunas filas de la nueva columna

    # Cargar el modelo de Sentence Transformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    while True:
        query = input('Ingresa el término de búsqueda (o escribe "salir" para terminar): ')
        if query.lower() == 'salir':
            print("Saliendo del programa. ¡Hasta luego!")
            break

        # Crear embeddings para los títulos o descripciones (ajusta la columna según el dataset)
        embeddings = model.encode(df['Description'].tolist(), show_progress_bar=True)

        # Crear embeddings para la consulta
        query_embedding = model.encode(query)

        # Calcular la similitud coseno entre la consulta y los embeddings de los títulos
        similarities = util.cos_sim(query_embedding, embeddings)

        # Agregar la similitud al DataFrame
        df['similarity'] = similarities.flatten()

        # Ordenar el DataFrame por similitud
        df_sorted = df.sort_values(by='similarity', ascending=False)

        # Mostrar los 5 resultados más similares
        print("Resultados más similares:")
        print(df_sorted[['Title', 'Cast', 'Info']].head())

if __name__ == '__main__':
    main()