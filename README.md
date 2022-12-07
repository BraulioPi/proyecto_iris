# proyecto_iris
Instrucciones para ejecución:
* Solo necesitas posicionarte en la terminal en la carpeta app
* Ejecutar el archivo .sh   *start.sh* (de preferencia sudo ./start.sh )
* Listo, este archivo genera todo e inicializa el programa.
#### El archivo .sh tiene un orden de ejecución:
* generate_iris: Genera la base de datos iris necesaria para todo.
* Corre el docker-compose para contruir las imagenes de cada servicio y levanta los contenedores.
* Tarda aproximandamente 4 minutos en construir y levantar todo, así que un poquito de paciencia.
* Una vez construidas las imagenes y levantado el docker-compose puedes abrir el dashboard de shiny poniendo en tu navegador : http://0.0.0.0:4999
* Si te aburres de utilizar el dashboard, solamente dale control + c en la terminal donde ejecutaste el .sh y se cerrará el servicio y se eliminará el ambiente virtual (si diste la opción que si).
