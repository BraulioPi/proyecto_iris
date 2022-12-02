FROM python:3.8
LABEL maintainer "Brauchi miaau"
COPY /iris.csv /app/iris.csv
COPY requirements.txt / 
RUN pip install -r /requirements.txt
COPY ./ ./
EXPOSE 4999
CMD ["shiny","run","--port", "4999","./primer_modelo.py"]