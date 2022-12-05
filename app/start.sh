#!/bin/bash
echo "Esta aplicación requiere librerias de python para generar los datos IRIS en su primer fase 
      Solo se instalarán en la primer fase: 
      -Pandas 
      -Numpy 
      -Sklearn 
      ¿Quieres crear un ambiente virtual de python para su instalación?
      Si no quieres, estas 3 librerias se instalaran en tu entorno global
      Si le das que si, se generara el entorno virtual y una vez que se genere los
      datos iniciales, ese entorno se eliminará de tu máquina.(RECOMENDADO)
      Responde con  y/n"
read respuesta
if [ $respuesta = "y" ]; then
    echo "creando ambiente virtual de python"
    python -m venv .venv 
    source .venv/bin/activate
    echo "ambiente virtual creado"
fi
pip install scikit-learn
pip install pandas
pip install numpy
python generate_iris.py
docker-compose up -d

if [ $respuesta = "y" ]; then 
    deactivate 
    rm -rf .venv
fi
