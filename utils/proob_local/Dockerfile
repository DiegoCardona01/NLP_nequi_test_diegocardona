# Usa la imagen base de Python
FROM python:3.12-slim

# Instala dependencias necesarias
RUN apt-get update && apt-get install -y libgomp1

# Establece el directorio de trabajo
WORKDIR /code

# Copia el archivo de requisitos
COPY requirements.txt .

# Instala las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia todo el contenido de la carpeta 'app' al directorio /app en el contenedor
COPY app/ ./app

COPY model_local/ ./model_local

COPY main.py .

# Copia el script 'initializer.sh' al directorio /app en el contenedor
COPY initializer.sh .

# Da permisos de ejecución al script 'initializer.sh'
RUN chmod +x initializer.sh

# Expone el puerto 8000
EXPOSE 8000

# Define el punto de entrada
ENTRYPOINT ["./initializer.sh"]
