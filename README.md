# Dashboard interactivo para la clasificación de especies de flores
El proyecto consiste en la implementación de un paquete de datos completo para la clasificación de flores. Mismo que está compuesto por una base de datos CRUD en *postgreSQL*, una API, y un desarrollada en python *shiny* que te permite interactuar con los datos y con el modelo. Para garantizar que sea reproducible, está todo dentro de un contenedor *docker*. 
## Datos
Utilizamos la versión de sklearn del conjunto de datos más famoso de la literatura de reconocimiento de patrones. Fue creado por el biólogo y estatista Sir Ronald Aylmer Fisher en 1936, para mostrar las diferencias morfológicas entre 3 especies de Iris e introducir una función lineal para clasificarlas, con Iris Setosa fácilmente separable.\
El dataset tiene 150 instancias, cada una con 4 atributos constituidos por las medidas de las dimensiones de los sépalos y los pétalos de cada flor. Todos los datos están etiquetados. \
*variables*
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm \
![Medidas de sépalo y pétalo](/codigo_usado_desarrollo/iris.png)\
Medidas de sépalo y pétalo de cada flor.
### Clasificación
- Iris Setosa
- Iris Versicolour
- Iris Virginica
## Modelo
Se implementó un modelo XGBoost, que fue elegido gracias a su buen desempeño, alcanzando las métricas
| **Métricas**||
| - | - |
| Accuracy | 0.97 |
| Precision | 0.96 |
| Recall | 0.96 |
| f1-score | 0.96 |
## Dashboard
El dashboard permite explorar e interactuar con la base de datos y con el modelo, ajustando su especificidad, mediante la modificación de los hiperparámetros:
- Profundidad máxima de cada árbol (max_depth)
- Peso mínimo (hessiano) requerido para cada hijo (min_child_weight)
- Número de árboles potenciados por gradiente (n_estimators)
### Ejecución
1. Posicionarse en la terminal en la carpeta app
2. Ejecutar el archivo .sh   *start.sh*\
(de preferencia sudo ./start.sh )
3. Listo, este archivo genera todo e inicializa el programa.
#### Orden de ejecución del archivo .sh
- generate_iris: Genera la base de datos iris necesaria para todo.
- Corre el docker-compose para contruir las imagenes de cada servicio y levanta los contenedores.
Tarda aproximandamente 4 minutos en construir y levantar todo, así que un poquito de paciencia.
- Una vez construidas las imagenes y levantado el docker-compose puedes abrir el dashboard de shiny poniendo en tu navegador : http://0.0.0.0:4999
- Si te aburres de utilizar el dashboard, control + c en la terminal donde ejecutaste el .sh y se cerrará el servicio y se eliminará el ambiente virtual (si diste la opción que si).
