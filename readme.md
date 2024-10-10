# Buscador de Películas Semántico

Este programa permite realizar búsquedas en un dataset de las 1000 mejores películas de IMDB utilizando embeddings de texto para encontrar coincidencias relevantes basadas en descripciones.

## Requisitos

Antes de ejecutar el programa, asegúrese de tener instaladas las siguientes librerias de Python. Puede gestionar la instalación desde el archivo `requerimientos.txt`.

- pandas
- tqdm
- sentence-transformers

## Descarga e Instalación

Sigue estos pasos para descargar e instalar el programa en tu máquina local:

1. **Copia el repositorio o descarga los archivos**:
   - Si usa Git, puedes clonar el repositorio con el siguiente comando:
     ---bash
     git clone <https://github.com/crandega/Peliculas.git>
     ---
   - Tambien puede descargar directamente el archivo `main.py`, el archivo `IMDB top 1000.csv`, y `requirements.txt` en una carpeta de su elección.

2. **Navega a la carpeta del proyecto**:
   Abre una terminal e ingresa a la carpeta donde descargaste los archivos.

   Crea un entorno virtual (opcional): Para evitar conflictos entre paquetes de diferentes proyectos.
      ---bash
   python -m venv env
   #Activa el entorno virtual:
   .\env\Scripts\activate
   ---

   Instala las dependencias: Utiliza el archivo requerimientos.txt para instalar las librerias necesarias.
      ---bash
   pip install -r requerimientos.txt
   ---

   Ejecución del Programa
   Una vez instalado todas las dependencias, puede ejecutar el programa con el siguiente comando:

      ---bash
   python main.py

## Uso
El programa pedirá que ingreses un término de búsqueda.
Puedes buscar películas con una descripción.
Para salir del programa, escribe salir y presiona Enter.

## Notas
Asegurese que el archivo IMDB top 1000.csv esté en la misma carpeta que main.py para que el programa funcione correctamente.