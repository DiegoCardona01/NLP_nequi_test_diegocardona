FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos
COPY appi/requirements-api.txt .

# Instala las dependencias
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Copia todo el contenido de appi/ al directorio /app en el contenedor
COPY appi/ .

# Copia el script initializer.sh al directorio /app en el contenedor
COPY initializer.sh .

# Da permisos de ejecución al script
RUN chmod +x initializer.sh

# Expone el puerto 8000
EXPOSE 8000

# Define el punto de entrada
ENTRYPOINT ["./initializer.sh"]