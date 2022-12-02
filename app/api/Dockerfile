FROM continuumio/miniconda3
EXPOSE 8080
COPY ./environment.yml /app/environment.yml
RUN conda env update -f /app/environment.yml
RUN activate proyecto_iris
COPY . /app
ENTRYPOINT [ "conda","run","--no-capture-output","-n","proyecto_iris","python","/app/app.py" ]