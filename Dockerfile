# Usar la imagen base oficial de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos al directorio de trabajo en el contenedor
COPY requirements.txt /app/requirements.txt

# Instalar las dependencias
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el resto del código fuente del proyecto al contenedor
COPY src ./src/
COPY tests /app/tests

# Añadir el directorio src al PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Variable de entorno para cambiar entre el modo de aplicación y pruebas
ENV RUN_MODE=app

# Comando para ejecutar la aplicación principal o las pruebas unitarias
CMD if [ "$RUN_MODE" = "test" ]; then \
        python -m unittest discover -s /app/tests; \
    else \
        python /app/src/buscar_peliculas.py; \
    fi