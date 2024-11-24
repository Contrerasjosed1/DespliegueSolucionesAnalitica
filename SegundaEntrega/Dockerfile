# Usar una imagen base de Python 3.10
FROM python:3.10

# Crear un usuario para ejecutar la aplicación
RUN adduser --disabled-password --gecos '' dash-user

# Establecer el directorio de trabajo
WORKDIR /opt/app

# Copiar los archivos de la aplicación
ADD . /opt/app

# Instalar las dependencias necesarias
RUN pip install --upgrade pip
RUN pip install -r /opt/app/requirements.txt

# Dar permisos de ejecución a los scripts necesarios
RUN chmod +x /opt/app

# Cambiar la propiedad de los archivos al usuario creado
RUN chown -R dash-user:dash-user /opt/app

# Cambiar al usuario creado
USER dash-user

# Exponer el puerto 8050 para el tablero
EXPOSE 8050

# Definir variables de entorno por defecto
ENV PORT=8050
ENV API_URL=http://localhost:8001/api/v1/predict

# Comando para ejecutar la aplicación
CMD ["python", "app_1.py"]