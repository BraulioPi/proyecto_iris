FROM python:3.8
LABEL maintainer "Brauchi miaau"
RUN mkdir -p /app/images
WORKDIR /app
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
COPY ./ ./
EXPOSE 4999
CMD ["shiny","run","--host", "0.0.0.0","--port", "4999","./primer_modelo.py"]