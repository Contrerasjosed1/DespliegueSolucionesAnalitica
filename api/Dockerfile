# Usar una imagen base de Python 3.10
FROM python:3.11

# Crear un usuario no privilegiado para ejecutar la aplicación
RUN adduser --disabled-password --gecos '' api-user

# Establecer el directorio de trabajo
WORKDIR /opt/api

# Copiar los archivos de la aplicación
ADD . /opt/api

# Instalar las dependencias necesarias
RUN pip install --upgrade pip
RUN pip install -r /opt/api/requirements.txt

# Asegurarse de que el script de inicio tenga permisos de ejecución
RUN chmod +x /opt/api/run.sh

# Cambiar la propiedad de los archivos al usuario creado
RUN chown -R api-user:api-user /opt/api

# Cambiar al usuario no privilegiado
USER api-user

# Exponer el puerto de la API
EXPOSE 8001

# Comando para ejecutar la aplicación
CMD ["bash", "./run.sh"]